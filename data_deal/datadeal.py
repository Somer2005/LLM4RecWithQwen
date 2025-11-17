import pandas as pd
import os
import re
import numpy as np
from urllib.request import urlretrieve
import zipfile
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

class Movielens(Dataset):
    def __init__(self, ratings_path='data/ratings.csv', 
                       movies_path='data/movies.csv', 
                       users_path='data/users.csv'):
        super().__init__()
        # 1. 读取原始表
        ratings = pd.read_csv(ratings_path)
        movies = pd.read_csv(movies_path)
        users = pd.read_csv(users_path)

        # 2. 预处理 movies
        movies['clean_title'] = movies['title'].apply(clean_text)
        genre_columns = ['unknown', 'Action', 'Adventure', 'Animation',
                         'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
        for col in genre_columns:
            if col not in movies.columns:
                movies[col] = 0   # 若缺失则补 0
        movies['genres'] = movies[genre_columns].apply(lambda row: ','.join([g for g in genre_columns if row[g]==1]), axis=1)
        self.movies = movies[['movieId','title','clean_title','genres']]

        # 3. 预处理 users
        users['gender_code'] = LabelEncoder().fit_transform(users['gender'])
        users['occupation_code'] = LabelEncoder().fit_transform(users['occupation'])
        self.users = users[['userId','age','gender_code','occupation_code']]

        # 4. 保存 ratings
        self.ratings = ratings.reset_index(drop=True)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, index):
        row = self.ratings.iloc[index]
        user_id = row['userId']
        movie_id = row['movieId']

        # 获取用户特征
        user_features = self.users[self.users['userId']==user_id].iloc[0].to_dict()
        # 获取电影特征
        movie_features = self.movies[self.movies['movieId']==movie_id].iloc[0].to_dict()

        # 返回一个完整字典
        return {
            "userId": user_id,
            "movieId": movie_id,
            "rating": row['rating'],
            "timestamp": row['timestamp'],
            "user_features": user_features,
            "movie_features": movie_features
        }
    

class MLDataLoader(DataLoader):
  def __init__(self, dataset, batch_size=256, shuffle=True):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )



#下面这个函数做的事情主要是：清理干净文本，其实主要的操作对象就是电影标题啦。
STOP_WORDS = {
    "the", "and", "a", "an", "of", "in", "on", "for", "with", "to", "at", "by", 
    "from", "up", "about", "as", "is", "it", "this", "that", "be", "are", "was", "were"
}

def clean_text_no_nltk(text):
    # 去掉年份
    text = re.sub(r'\(\d{4}\)', '', text)
    # 去掉特殊字符，只保留字母和数字
    text = re.sub(r'[^\w\s]', '', text)
    # 小写化
    text = text.lower()
    # 简单分词：按空格切分
    tokens = text.split()
    # 过滤停用词和长度 <=2 的词
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

def build_user_preferences(ratings, movies):
    
    
    def get_user_preferences(user_id, top_n=3):
        
        user_ratings = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] >= 4)]
        if len(user_ratings) == 0:
            return ['Drama', 'Comedy']  
        #这里返回了一个默认偏好，其实你爱选啥选啥无所谓这个
        
        user_items = user_ratings['item_id'].unique()
        user_movies = movies[movies['item_id'].isin(user_items)]
        
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
    user = users[users['user_id'] == user_id].iloc[0]
    
    age_group = 'young' if user['age'] < 30 else 'middle-aged' if user['age'] < 50 else 'senior'
    gender = 'male' if user['gender_code'] == 1 else 'female'
    
    return f"""Recommend 5 movies for a {age_group} {gender} who likes {', '.join(user_prefs)} movies. 
    Only return the movie titles, one per line, without any additional text or numbering."""

def main():
    print("加载和处理数据...")
    dataset = Movielens()  # 初始化时已经做了预处理

    # 保存处理后的数据（可选）
    dataset.ratings.to_csv('data/ratings_processed.csv', index=False)
    dataset.movies.to_csv('data/movies_processed.csv', index=False)
    dataset.users.to_csv('data/users_processed.csv', index=False)

    print("数据准备完成！")


if __name__=="__main__":
    main()
