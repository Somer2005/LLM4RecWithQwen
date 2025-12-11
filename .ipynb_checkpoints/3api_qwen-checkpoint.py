import json
import time
import os
from openai import OpenAI
import pandas as pd

class LLMApiClient:
    def __init__(self, api_key, base_url, model_name):
        """
        初始化 API 客户端
        :param api_key: 你的 API 密钥
        :param base_url: API 的地址 (例如 DeepSeek, 通义千问, 或 OpenAI 官方地址)
        :param model_name: 模型名称 (例如 "deepseek-chat", "qwen-plus")
        """
        self.client = OpenAI(
            api_key="*******",
            base_url="******"
        )
        self.model_name = "*******"
        print(f"API 客户端初始化完成，使用模型: {self.model_name}")

    def generate(self, prompt, max_tokens=500, temperature=0.7):
        #调用远程 API 生成文本
        start_time = time.time()
            
        # 构造符合 OpenAI 标准的消息格式
        messages = [
            {"role": "system", "content": "你是一个专业的电影推荐助手。"},
            {"role": "user", "content": prompt}
        ]

        # 发起请求
        response = self.client.chat.completions.create(
        model=self.model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False # 这里设为 False 简化处理，也可以设为 True 做流式输出
        )
            
            # 提取结果
        content = response.choices[0].message.content
        print(f"API 生成完成，耗时: {time.time() - start_time:.2f}秒")
        return content
    
    def get_recommendations(self, user_id, build_prompt_func, users, user_preferences_func):
        prompt = build_prompt_func(user_id, users, user_preferences_func)
        print("生成的提示词（前100字符）：", prompt[:100] + "...")
        
        response = self.generate(prompt)
        if not response:
            return []
            
        recommendations = [line.strip() for line in response.split('\n') if line.strip()]
        # 过滤掉可能的废话，只保留看起来像电影名的行（根据实际返回情况调整）
        return recommendations[:5]

# ================= 主程序调用部分 =================

if __name__ == "__main__":
    try:
        # 配置 API 信息 
        
        API_KEY = "*****" 
        BASE_URL = "****" 
        MODEL_NAME = "***********"        
        client = LLMApiClient(
            api_key=API_KEY,
            base_url=BASE_URL,
            model_name=MODEL_NAME
        )
        
        # 数据加载保持不变
        users = pd.read_csv('/root/autodl-tmp/LLM4RecWithQwen/data/users.csv')
        movies_processed = pd.read_csv('/root/autodl-tmp/LLM4RecWithQwen/data/movies.csv')
        ratings_processed = pd.read_csv('/root/autodl-tmp/LLM4RecWithQwen/data/ratings.csv')
        
        # 假定这两个函数存在于你的 data_preparation.py 中
        from data_preparation import build_user_preferences, build_prompt
        user_prefs_func = build_user_preferences(ratings_processed, movies_processed)
        
        print("测试 API 推荐结果:")
        recs = client.get_recommendations(
            user_id=42,
            build_prompt_func=build_prompt,
            users=users,
            user_preferences_func=user_prefs_func
        )
        
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec}")
            