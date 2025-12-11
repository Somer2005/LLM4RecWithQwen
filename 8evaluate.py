import pandas as pd
import numpy as np
import time
from sklearn.metrics import ndcg_score
import traceback
from recommender import init_system 
from data_preparation import build_prompt
from recall_modules import merge_recalls

def evaluate_recall(user_id, recall_func, relevant_items, k=10):
    """评估召回效果"""
    if not relevant_items:
        return None
        
    start_time = time.time()
    try:
        # 调用传入的 recall_func (即下文定义的 wrapper)
        # 返回格式: [(item_id, score), ...]
        recalled_items = [item_id for item_id, _ in recall_func(user_id, top_k=k)]
    except Exception as e:
        print(f"用户 {user_id} 召回失败: {str(e)}")
        # traceback.print_exc() # 调试时可打开
        return None
    latency = time.time() - start_time
    
    hits = len(set(recalled_items) & relevant_items)
    # 召回率 = 召回且相关的 / 所有相关的
    recall_rate = hits / len(relevant_items) if relevant_items else 0
    # 准确率 = 召回且相关的 / 所有召回的
    precision = hits / len(recalled_items) if recalled_items else 0
    
    return {
        'user_id': user_id,
        'recall_rate': recall_rate,
        'precision': precision,
        'hits': hits,
        'latency': latency
    }

def evaluate_ranking(user_id, rank_func, user_ratings, candidates, k=10):
    """评估排序效果"""
    # 如果用户评分过少，或者没有候选项，不进行排序评估
    if len(user_ratings) < 5 or not candidates:
        return None
        
    start_time = time.time()
    try:
        # rank_func 返回排序后的 [(item_id, score), ...]
        ranked_items = [item_id for item_id, _ in rank_func(user_id, candidates)[:k]]
    except Exception as e:
        print(f"用户 {user_id} 排序失败: {str(e)}")
        return None
    latency = time.time() - start_time
    
    # 构建真实相关性列表 (Ground Truth)
    relevance = []
    item_rating_map = dict(zip(user_ratings['item_id'], user_ratings['rating']))
    
    for item_id in ranked_items:
        # 如果模型推荐了用户没看过的电影，相关性视为 0 (严格模式)
        # 或者是使用整个测试集的平均分作为填充
        relevance.append(item_rating_map.get(item_id, 0))
    
    if sum(relevance) == 0:
        return None
        
    # 计算 NDCG
    # 理想排序：按真实评分从高到低
    ideal_relevance = sorted(relevance, reverse=True) 
    
    # 如果理想排序全是0，NDCG无意义
    if sum(ideal_relevance) == 0:
        return None

    ndcg = ndcg_score([relevance], [ideal_relevance])
    
    return {
        'user_id': user_id,
        'ndcg': ndcg,
        'latency': latency
    }

def run_evaluation(model_name="qwen2-7b-instruct", sample_size=20, k=10):
    """运行完整评估（适配新版接口）"""
    print(f"====== 开始评估推荐系统 ======")
    print(f"模型: {model_name} | 样本数: {sample_size} | Top K: {k}")
    
    try:
        # 1. 初始化系统
        system = init_system() # 注意：新版 init_system 可能不需要参数，或者根据你的实现传递
        if system is None:
            print("系统初始化失败，无法进行评估")
            return None
        
        # 2. 解包组件 (根据 app.py 的 return 结构)
        ratings = system['ratings']
        users = system['users']
        movies = system['movies']
        
        # 获取召回器
        two_tower_recaller = system['two_tower_recaller'] # 修改：原 dssm_recaller
        keyword_recaller = system['keyword_recaller']
        llm_recaller = system['llm_recaller']
        
        ranker = system['ranker']
        user_preferences_func = system['user_preferences_func']
        llm_client = system['llm_client'] # 修改：获取 OpenAI client
        
        # 3. 定义辅助函数 (闭包)
        
        # (A) 定义 LLM 生成逻辑 (与 app.py 保持一致)
        def llm_generate_func(uid):
            user_row = users[users['user_id'] == uid]
            if user_row.empty: return []
            
            user_data = user_row.iloc[0]
            # 构建 Prompt
            prompt = build_prompt(user_data, user_preferences_func(uid))
            
            try:
                response = llm_client.chat.completions.create(
                    model=system.get('model_name', model_name),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=200
                )
                content = response.choices[0].message.content
                return [line.strip() for line in content.split('\n') if len(line.strip()) > 2]
            except Exception as e:
                print(f"[Eval LLM Error] User {uid}: {e}")
                return []

        # (B) 封装 Recall 流程
        def recall_func(user_id, top_k=10):
            # 获取用户画像文本
            user_prefs = user_preferences_func(user_id)
            user_profile_text = ", ".join(user_prefs) if user_prefs else ""
            
            return merge_recalls(
                user_id=user_id,
                two_tower_recaller=two_tower_recaller, # 传入双塔
                keyword_recaller=keyword_recaller,
                llm_recaller=llm_recaller,
                ratings_df=ratings,
                llm_generate_func=llm_generate_func,   # 传入生成函数
                user_profile_text=user_profile_text,   # 传入画像文本
                top_k=top_k,
                weights={'two_tower': 0.5, 'keyword': 0.1, 'llm': 0.35, 'ad': 0.05} # 保持权重一致
            )
        
        # (C) 封装 Ranking 流程
        def rank_func(user_id, candidates):
            return ranker.rank(
                user_id=user_id,
                candidate_items=candidates,
                users_df=users,
                ratings_df=ratings,
                return_scores=True 
            )
        
        # 4. 选择测试用户
        # 筛选交互数 >= 20 的活跃用户，以保证 ground truth 充足
        valid_users = [
            uid for uid in ratings['user_id'].unique()
            if len(ratings[ratings['user_id'] == uid]) >= 20
        ]
        
        if len(valid_users) < sample_size:
            print(f"提示：有效用户({len(valid_users)})少于请求样本数({sample_size})，使用所有有效用户。")
            sample_size = len(valid_users)
            
        sample_users = np.random.choice(valid_users, size=sample_size, replace=False)
        
        # 5. 执行评估循环
        recall_results = []
        ranking_results = []
        total_latency = 0
        
        print("\n正在逐个评估用户...")
        for i, user_id in enumerate(sample_users, 1):
            if i % 5 == 0: print(f"进度: {i}/{sample_size}")
            
            # Ground Truth: 用户实际评分 >= 4 的电影
            user_ratings = ratings[ratings['user_id'] == user_id].copy()
            relevant_items = set(user_ratings[user_ratings['rating'] >= 4]['item_id'])
            
            # --- 评估召回 ---
            recall_res = evaluate_recall(user_id, recall_func, relevant_items, k)
            if recall_res:
                recall_results.append(recall_res)
                total_latency += recall_res['latency']
            
            # --- 评估排序 ---
            # 扩大召回池给排序模型 (例如 Top-50)
            candidates = recall_func(user_id, top_k=50) 
            if not candidates:
                continue
                
            rank_res = evaluate_ranking(user_id, rank_func, user_ratings, candidates, k)
            if rank_res:
                ranking_results.append(rank_res)
                total_latency += rank_res['latency']
        
        # 6. 统计结果
        avg_recall = np.mean([r['recall_rate'] for r in recall_results]) if recall_results else 0
        avg_precision = np.mean([r['precision'] for r in recall_results]) if recall_results else 0
        avg_ndcg = np.mean([r['ndcg'] for r in ranking_results]) if ranking_results else 0
        
        total_evals = len(recall_results) + len(ranking_results)
        avg_latency = total_latency / total_evals if total_evals > 0 else 0
        
        print("\n" + "="*30)
        print("       最终评估结果       ")
        print("="*30)
        print(f"评估样本数  : {sample_size}")
        print(f"平均响应时间: {avg_latency:.4f} 秒")
        print("-" * 30)
        print(f"Recall@{k}    : {avg_recall:.4f}")
        print(f"Precision@{k} : {avg_precision:.4f}")
        print(f"NDCG@{k}      : {avg_ndcg:.4f}")
        print("="*30 + "\n")
        
        return {
            'avg_recall': avg_recall,
            'avg_precision': avg_precision,
            'avg_ndcg': avg_ndcg,
            'avg_latency': avg_latency
        }
    
    except Exception as e:
        print(f"评估过程发生严重错误: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 记得根据你的实际模型名称修改这里
    run_evaluation(model_name="qwen2-7b-instruct", sample_size=5, k=10)