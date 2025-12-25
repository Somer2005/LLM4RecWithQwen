import pandas as pd
import os
import re
import numpy as np
from urllib.request import urlretrieve
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
import nltk

# 按字段，读取并整理数据集，数据集包括物品ID，标题，发布时间，电影上映时间，URL，电影种类
def load_dataset():
    data_path = '/root/autodl-tmp/LLM4RecWithQwen/data'
    ratings = pd.read_csv(
        f'{data_path}/u.data', 
        sep='\t', 
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    
    movies = pd.read_csv(
        f'{data_path}/u.item', 
        sep='|', 
        names=['item_id', 'title', 'release_date', 'video_release_date', 
               'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
               'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
               'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
               'Thriller', 'War', 'Western'], 
        encoding='latin-1'
    )
    
    users = pd.read_csv(
        f'{data_path}/u.user', 
        sep='|', 
        names=['user_id', 'age', 'gender', 'occupation', 'zipcode']
    )
    
    return ratings, movies, users

# 清理掉异常值
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    
    
    text = re.sub(r'\(\d{4}\)', '', text) 
    text = re.sub(r'[^\w\s]', '', text)  
    tokens = word_tokenize(text.lower()) 
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]  # 过滤
    return ' '.join(tokens)

# 数据处理：最开始的数据是One-Hot的，然后我们要做的事情就是
def preprocess_data(ratings, movies, users):
    
    movies['clean_title'] = movies['title'].apply(clean_text)
    genre_columns = ['unknown', 'Action', 'Adventure', 'Animation',
                   'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                   'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                   'Thriller', 'War', 'Western']
    
    missing_columns = [col for col in genre_columns if col not in movies.columns]
    if missing_columns:
        raise ValueError(f"电影数据缺少类型列: {missing_columns}")
    
    def get_genres(row):
        return ', '.join([col for col in genre_columns if row[col] == 1])
    movies['genres'] = movies.apply(get_genres, axis=1)
    
    if 'genres' not in movies.columns:
        raise ValueError("genres字段未成功生成")
    if movies['genres'].isna().all():
        raise ValueError("genres字段全为空，请检查类型列处理逻辑")
    
    movies = movies[['item_id', 'title', 'clean_title', 'genres']]
    
    users['gender_code'] = LabelEncoder().fit_transform(users['gender'])
    users['occupation_code'] = LabelEncoder().fit_transform(users['occupation'])
    users = users[['user_id', 'age', 'gender_code', 'occupation_code']]
    
    return ratings, movies, users
#上面这个数据做好了类别的One-Hot
#下面我们继续
#下面这个函数用来构建用户画像，我们首先设置了一个user_ratings变量，而后，首先是我们假设如果用户没打过分，那么就默认是Drama，Comedy，
def build_user_preferences(ratings, movies):
    
    
    def get_user_preferences(user_id, top_n=3):
        
        user_ratings = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] >= 4)]
        #冷启动兜底策略：因为我们的MovieLens-100K数据集中，大家的偏好很多都是Drama，Comedy，因此我们设定为comedy
        if len(user_ratings) == 0:
            return ['Drama', 'Comedy']  # 默认偏好
        #构建一个用户喜欢的item塔
        user_items = user_ratings['item_id'].unique()
        user_movies = movies[movies['item_id'].isin(user_items)]
        #构建一个用户偏好哈希，这个哈希做的事情是：统计用户喜欢的类别数目。
        genre_counts = {}
        for _, row in user_movies.iterrows():
            for genre in row['genres'].split(', '):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        if not genre_counts:
            return ['Drama', 'Comedy']
        return [genre for genre, _ in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    
    return get_user_preferences

def build_prompt(user_id, users, get_user_preferences):
    """为用户构建推荐提示词"""
    user_prefs = get_user_preferences(user_id)
    #确定好我们用户最多的种类
    user = users[users['user_id'] == user_id].iloc[0]
    #利用好年龄和性别特征
    age_group = 'young' if user['age'] < 30 else 'middle-aged' if user['age'] < 50 else 'senior'
    gender = 'male' if user['gender_code'] == 1 else 'female'
    
    return f"""5 movies. Profile: {age_group} {gender}, likes {', '.join(user_prefs)}.Output: only titles, one per line."""

def main():
    print("加载数据...")
    ratings, movies, users = load_dataset()
    
    print("预处理数据...")
    ratings, movies, users = preprocess_data(ratings, movies, users)
    
    ratings.to_csv('/root/autodl-tmp/LLM4RecWithQwen/data/ratings.csv', index=False)
    movies.to_csv('/root/autodl-tmp/LLM4RecWithQwen/data/movies.csv', index=False)
    users.to_csv('/root/autodl-tmp/LLM4RecWithQwen/data/users.csv', index=False)
    
    print("数据准备完成！")

if __name__ == "__main__":
    main()


