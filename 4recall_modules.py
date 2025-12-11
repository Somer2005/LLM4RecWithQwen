import pandas as pd
import numpy as np
import re
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- 全局配置 ---
DATA_DIR = "/root/autodl-tmp/LLM4RecWithQwen/data"
EMB_DIR = os.path.join(DATA_DIR, "embeddings")
nltk.data.find('tokenizers/punkt')
nltk.data.find('corpora/stopwords')

def _normalize(scores_dict):
    """Min-Max 归一化"""
    if not scores_dict:
        return {}
    values = list(scores_dict.values())
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return {k: 1.0 for k in scores_dict}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}


# --- 1. 双塔召回 ---
class TwoTowerRecaller:
    def __init__(self):
        self.user_embs = None
        self.item_embs = None
        self.user_id_map = {}
        self.movie_id_map = {}
        self.fitted = False

    def load_model(self):
        print("正在加载 DSSM 双塔向量...")
        try:
            self.user_embs = np.load(os.path.join(EMB_DIR, "user_tower_embeddings.npy"))
            self.item_embs = np.load(os.path.join(EMB_DIR, "item_tower_embeddings.npy"))
            self.user_ids = np.load(os.path.join(EMB_DIR, "user_ids.npy"))
            self.movie_ids = np.load(os.path.join(EMB_DIR, "movie_ids.npy"))
            
            self.user_id_map = {str(uid): idx for idx, uid in enumerate(self.user_ids)}
            self.movie_id_map = {idx: int(mid) for idx, mid in enumerate(self.movie_ids)}
            
            self.fitted = True
            print(f"DSSM 加载完毕: User {self.user_embs.shape}, Item {self.item_embs.shape}")
        except FileNotFoundError:
            print(f"[Warning] 找不到双塔模型文件于 {EMB_DIR}")
            self.fitted = False

    def recall(self, user_id, top_k=20):
        if not self.fitted: return []
        u_key = str(user_id)
        if u_key not in self.user_id_map: return []
        
        user_idx = self.user_id_map[u_key]
        u_vec = self.user_embs[user_idx]
        scores = np.dot(self.item_embs, u_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.movie_id_map[idx], float(scores[idx])) for idx in top_indices]


# --- 2. 关键词召回 ---
class KeywordRecaller:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        self.tfidf_matrix = None
        self.movie_ids = None
        self.movies_df = None
        self.fitted = False
        
    def fit(self, movies_df):
        self.movies_df = movies_df
        self.movie_ids = movies_df['item_id'].values
        texts = movies_df['clean_title'].fillna('') + ' ' + movies_df['genres'].fillna('')
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.fitted = True
        return self
    
    def recall(self, query, top_k=10):
        if not self.fitted: return []
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]
        return [(self.movie_ids[i], float(similarities[i])) for i in top_indices]


# --- 3. 热门召回 (兜底) ---
def ad_recall(ratings_df, top_k=5):
    if ratings_df is None or ratings_df.empty:
        return []
    pop = ratings_df.groupby('item_id').agg({'rating': ['count', 'mean']}).reset_index()
    pop.columns = ['item_id', 'cnt', 'mean']
    max_cnt = pop['cnt'].max() if pop['cnt'].max() > 0 else 1
    pop['score'] = (pop['cnt'] / max_cnt * 0.7) + (pop['mean'] / 5.0 * 0.3)
    res = pop.sort_values('score', ascending=False).head(top_k)
    return [(int(row['item_id']), float(row['score'])) for _, row in res.iterrows()]


# --- 4. LLM 召回 ---
class LLMRecaller:
    def __init__(self, movies_df, vectorizer=None):
        self.movies_df = movies_df
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        self.tfidf_matrix = None
        self.movie_ids = None
        self.ratings_df = None
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.stop_words = set()

        self._fit_vectorizer()
        
    def _fit_vectorizer(self):
        texts = self.movies_df['clean_title'].fillna('') + ' ' + self.movies_df['genres'].fillna('')
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.movie_ids = self.movies_df['item_id'].values
    
    def _clean_text(self, text):
        text = str(text)
        text = re.sub(r'\(\d{4}\)', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        return ' '.join(tokens)
        
    def _find_similar_movies(self, text, top_k=5):
        clean_text = self._clean_text(text)
        if not clean_text.strip(): return []
        text_vec = self.vectorizer.transform([clean_text])
        similarities = cosine_similarity(text_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]
        return [(self.movie_ids[i], float(similarities[i])) for i in top_indices]
    
    def set_ratings_data(self, ratings_df):
        self.ratings_df = ratings_df

    def recall(self, user_id, llm_generate_func, top_k=10):
        # 1. 调用传入的函数获取标题
        try:
            generated_titles = llm_generate_func(user_id)
        except Exception:
            generated_titles = []
        
        # 2. 失败则兜底
        if not generated_titles:
            if self.ratings_df is not None:
                return ad_recall(self.ratings_df, top_k=top_k)
            return []
        
        # 3. 匹配
        all_matches = {}
        for title in generated_titles:
            matches = self._find_similar_movies(title, top_k=5)
            for item_id, score in matches:
                if score > all_matches.get(item_id, 0):
                    all_matches[item_id] = score
            
        return sorted(all_matches.items(), key=lambda x: x[1], reverse=True)[:top_k]


# --- 5. 融合逻辑 (重点修改了这里) ---
def merge_recalls(user_id, 
                  two_tower_recaller,  # <--- 新增
                  keyword_recaller, 
                  llm_recaller, 
                  ratings_df, 
                  llm_generate_func,   # <--- 新增
                  user_profile_text="", 
                  weights={'two_tower': 0.5, 'keyword': 0.1, 'llm': 0.35, 'ad': 0.05},
                  top_k=15,
                  return_details=False):
    
    # 1. 各路召回
    res_tt = dict(two_tower_recaller.recall(user_id, top_k=20))
    
    res_kw = {}
    if keyword_recaller.fitted and user_profile_text:
        res_kw = dict(keyword_recaller.recall(user_profile_text, top_k=20))
        
    res_llm = dict(llm_recaller.recall(user_id, llm_generate_func, top_k=20))
    res_ad = dict(ad_recall(ratings_df, top_k=10))
    
    # 2. 归一化
    norm_tt = _normalize(res_tt)
    norm_kw = _normalize(res_kw)
    norm_llm = _normalize(res_llm)
    norm_ad = _normalize(res_ad)
    
    # 3. 融合
    all_items = set(norm_tt.keys()) | set(norm_kw.keys()) | set(norm_llm.keys()) | set(norm_ad.keys())
    fused_scores = {}
    
    for item_id in all_items:
        score = (
            norm_tt.get(item_id, 0) * weights.get('two_tower', 0) +
            norm_kw.get(item_id, 0) * weights.get('keyword', 0) +
            norm_llm.get(item_id, 0) * weights.get('llm', 0) +
            norm_ad.get(item_id, 0) * weights.get('ad', 0)
        )
        fused_scores[item_id] = score
        
    # 4. 排序
    final_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    if return_details and keyword_recaller.movies_df is not None:
        id_map = keyword_recaller.movies_df.set_index('item_id')['title'].to_dict()
        return [(mid, score, id_map.get(mid, "Unknown")) for mid, score in final_results]
        
    return final_results