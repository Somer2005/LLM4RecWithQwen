import pandas as pd
import time
from flask import Flask, request, jsonify
import traceback
import os
from openai import OpenAI  # 使用 OpenAI SDK 调用兼容接口 (vLLM/DeepSeek/Qwen)

# 确保 recall_modules 已经是更新后的版本
from data_preparation import load_dataset, preprocess_data, build_user_preferences, build_prompt
from recall_modules import KeywordRecaller, LLMRecaller, TwoTowerRecaller, merge_recalls
from ranking_module import SimpleRanker
from kv_store import SimpleKVStore, CacheManager

app = Flask(__name__)

# --- 全局配置 ---
# 这里配置你的 LLM 服务地址 (例如 vLLM, DeepSeek, 或其他 OpenAI 兼容接口)
LLM_API_BASE = "http://localhost:8000/v1"  # 示例：本地 vLLM 地址
LLM_API_KEY = "EMPTY"  # 本地通常不需要 Key，如果是云服务则填写真实 Key
LLM_MODEL_NAME = "qwen2-7b-instruct" # 你的模型服务对应的模型名称

system_components = None
from threading import Lock
system_lock = Lock()

def init_system():
    """初始化推荐系统组件（线程安全）"""
    with system_lock:
        print("正在初始化推荐系统...")
        start_time = time.time()
        
        try:
            print("加载数据...")
            ratings, movies, users = load_dataset()
            ratings, movies, users = preprocess_data(ratings, movies, users)
            
            print("初始化缓存系统...")
            kv_store = SimpleKVStore()
            cache_manager = CacheManager(kv_store)
            
            print("构建用户偏好模型...")
            user_preferences_func = build_user_preferences(ratings, movies)
            
            # --- 改动 1: 初始化通用 LLM 客户端 (替代 Ollama) ---
            print(f"初始化 LLM 客户端 (Base: {LLM_API_BASE})...")
            llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
            
            # --- 改动 2: 初始化双塔召回 ---
            print("初始化双塔召回模块...")
            two_tower_recaller = TwoTowerRecaller()
            two_tower_recaller.load_model() # 加载 .npy 权重
            
            print("初始化关键词召回模块...")
            keyword_recaller = KeywordRecaller()
            keyword_recaller.fit(movies)
            
            # --- 改动 3: LLMRecaller 不再绑定 Client，只绑定数据 ---
            print("初始化 LLM 召回模块...")
            llm_recaller = LLMRecaller(movies_df=movies)
            llm_recaller.set_ratings_data(ratings) 
            
            print("初始化排序模块...")
            ranker = SimpleRanker()
            ranker.train(users, ratings)  
            
            print(f"系统初始化完成，耗时: {time.time() - start_time:.2f}秒")
            
            return {
                'ratings': ratings,
                'movies': movies,
                'users': users,
                'kv_store': kv_store,
                'cache_manager': cache_manager,
                'user_preferences_func': user_preferences_func,
                'llm_client': llm_client,          # 存储 OpenAI client
                'two_tower_recaller': two_tower_recaller, # 存储双塔实例
                'keyword_recaller': keyword_recaller,
                'llm_recaller': llm_recaller,
                'ranker': ranker,
                'model_name': LLM_MODEL_NAME,
                'initialized_at': time.time()
            }
            
        except Exception as e:
            print(f"系统初始化失败: {str(e)}")
            traceback.print_exc()
            return None

@app.route('/recommend', methods=['GET'])
def recommend():
    """推荐API接口"""
    global system_components
    
    # 确保系统已初始化
    with system_lock:
        if system_components is None:
            system_components = init_system()
            if system_components is None:
                return jsonify({'status': 'error', 'message': '系统初始化失败'}), 500
    
    try:
        user_id = int(request.args.get('user_id', 1))
        top_k = int(request.args.get('top_k', 10))
        if top_k < 1 or top_k > 50:
            return jsonify({'status': 'error', 'message': 'top_k必须在1-50之间'}), 400
        
        print(f"处理用户 {user_id} 的推荐请求 (top_k={top_k})...")
        
        # 解包组件
        c = system_components
        ratings, movies, users = c['ratings'], c['movies'], c['users']
        cache_manager = c['cache_manager']
        user_preferences_func = c['user_preferences_func']
        
        # 获取召回器实例
        two_tower_recaller = c['two_tower_recaller']
        keyword_recaller = c['keyword_recaller']
        llm_recaller = c['llm_recaller']
        ranker = c['ranker']
        llm_client = c['llm_client']
        
        # 1. 检查缓存
        cached_recs = cache_manager.get_cached_recommendations(user_id)
        if cached_recs:
            print(f"返回用户 {user_id} 的缓存推荐结果")
            return jsonify({
                'status': 'success',
                'source': 'cache',
                'user_id': user_id,
                'recommendations': cached_recs[:top_k]
            })
        
        # 2. 准备数据
        # 获取用户画像文本，用于关键词召回
        user_prefs_list = user_preferences_func(user_id)
        user_profile_text = ", ".join(user_prefs_list) if user_prefs_list else ""
        
        # --- 改动 4: 定义 LLM 生成函数 (闭包) ---
        # 这个函数将被传递给 merge_recalls 内部调用
        def llm_generate_func(uid):
            # 找到用户信息
            user_row = users[users['user_id'] == uid]
            if user_row.empty: return []
            user_data = user_row.iloc[0]
            
            # 构建 Prompt
            prompt = build_prompt(user_data, user_preferences_func(uid))
            
            # 调用 API
            response = llm_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            content = response.choices[0].message.content
            # 简单解析：假设按行分割
            titles = [line.strip() for line in content.split('\n') if len(line.strip()) > 2]
            return titles

        print(f"为用户 {user_id} 生成多路召回结果...")
        
        # --- 改动 5: 调用新的 merge_recalls ---
        candidates = merge_recalls(
            user_id=user_id,
            two_tower_recaller=two_tower_recaller, # 传入双塔实例
            keyword_recaller=keyword_recaller,
            llm_recaller=llm_recaller,
            ratings_df=ratings,
            llm_generate_func=llm_generate_func,   # 传入生成函数
            user_profile_text=user_profile_text,   # 传入用户画像文本
            weights={'two_tower': 0.5, 'keyword': 0.1, 'llm': 0.35, 'ad': 0.05}, # 自定义权重
            top_k=min(top_k * 4, 60) # 召回数量稍微放大一点给排序用
        )
        
        if not candidates:
            return jsonify({'status': 'error', 'message': '无法生成候选物品'}), 500

        # 3. 排序
        print(f"为用户 {user_id} 排序推荐结果...")
        ranked_items = ranker.rank(
            user_id=user_id,
            candidate_items=candidates,
            users_df=users,
            ratings_df=ratings,
            return_scores=True
        )
        
        # 4. 格式化输出
        recommendations = []
        movie_id_map = movies.set_index('item_id').to_dict('index')
        
        for item_id, score in ranked_items[:top_k]:
            if item_id in movie_id_map:
                movie = movie_id_map[item_id]
                recommendations.append({
                    'item_id': int(item_id),
                    'title': movie.get('title', 'Unknown'),
                    'genres': movie.get('genres', ''),
                    'score': float(score)
                })
        
        # 5. 写入缓存
        cache_manager.cache_recommendations(user_id, recommendations)
        
        return jsonify({
            'status': 'success',
            'source': 'computed',
            'user_id': user_id,
            'recommendations': recommendations,
            'model_used': LLM_MODEL_NAME
        })
        
    except Exception as e:
        print(f"推荐过程出错: {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """清除缓存接口"""
    global system_components
    if system_components is None:
        return jsonify({'status': 'error', 'message': '系统尚未初始化'}), 400
    
    try:
        data = request.json or {}
        user_id = data.get('user_id')
        clear_all = data.get('clear_all', False)
        
        if clear_all:
            system_components['kv_store'].clear_all()
            return jsonify({'status': 'success', 'message': '已清除所有缓存'})
        elif user_id is not None:
            success = system_components['cache_manager'].clear_user_cache(user_id)
            return jsonify({'status': 'success' if success else 'warning', 
                            'message': f'清除用户 {user_id} 缓存{"成功" if success else "失败"}'})
        else:
            count = system_components['cache_manager'].clear_expired_cache()
            return jsonify({'status': 'success', 'message': f'已清除 {count} 个过期缓存项'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    global system_components
    return jsonify({
        'status': "healthy" if system_components else "initializing",
        'timestamp': time.time(),
        'model_used': LLM_MODEL_NAME
    })

if __name__ == '__main__':
    print("启动推荐系统API服务...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, processes=1)