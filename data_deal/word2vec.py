import gensim
from gensim.models import Word2Vec
import logging
import os
import pandas as pd
import numpy as py 
import torch
import torch.nn as nn
import torch.optim as optim

class ContinuousBagofWords(nn.Module):
    def __init__(self,vocabulary_size,embedding_size,context_size):
        super(ContinuousBagofWords, self).__init__()
        #Continuous Bag of Words 这个东西，主要实现的功能是：将上下文embedding化后，根据softmax计算出中心词的概率分布
        self.embeddings=torch.nn.Embedding(vocabulary_size,embedding_size)

        self.linear=torch.nn.Linear(embedding_size,vocabulary_size)
        self.context_size=context_size

    def forward(self,inputs):
        # 1. 将输入的上下文索引转换为词向量
        #    输入: [batch_size, context_size]
        #    输出: [batch_size, context_size, embedding_dim]
        embeds = self.embeddings(inputs)
        
        # 2. 聚合上下文词向量
        #    这里我们使用求和的方式,将上下文合在一起
        #    dim=1 表示在 "context_size" 这个维度上求和
        #    输入: [batch_size, context_size, embedding_dim]
        #    输出: [batch_size, embedding_dim]
        context_vector = torch.sum(embeds, dim=1)
        
        # 3. 通过线性层得到预测分数
        #    输入: [batch_size, embedding_dim]
        #    输出: [batch_size, vocab_size]
        logits = self.linear(context_vector)
        return logits


class Skipgram(nn.Module):
    def __init__(self,vocabulary_size,embedding_size):
        super(Skipgram, self).__init__()
        #Skip-gram 这个东西，主要实现的功能是：将中心词embedding化后，根据softmax计算出上下文词的概率分布
        self.embeddings=torch.nn.Embedding(vocabulary_size,embedding_size)

        self.linear1=torch.nn.Linear(embedding_size,vocabulary_size)

    def forward(self,inputs):
        # 1. 将输入的中心词索引转换为词向量
        #    输入: [batch_size]
        #    输出: [batch_size, embedding_dim]
        embeds = self.embeddings(inputs)
        
        # 2. 通过线性层得到预测分数
        #    输入: [batch_size, embedding_dim]
        #    输出: [batch_size, vocab_size]
        logits = self.linear(embeds)
        return logits

df=pd.read_csv('data/movies.csv')
raw_text = """
We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
""".split()

vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

CONTEXT_SIZE = 2  # 左右各2个词
cbow_data = []
for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
    context = (
        [raw_text[i - 2], raw_text[i - 1]] +
        [raw_text[i + 1], raw_text[i + 2]]
    )
    target = raw_text[i]
    # 将词语转换为索引
    context_idxs = [word_to_ix[w] for w in context]
    target_idx = word_to_ix[target]
    cbow_data.append((context_idxs, target_idx))

print("CBOW 训练数据示例:", cbow_data[0])

# 我们选择训练 CBOW 模型
EMBEDDING_DIM = 10 # 词向量维度
model = ContinuousBagofWords(vocab_size, EMBEDDING_DIM)

# 定义损失函数
# nn.CrossEntropyLoss 内部已经包含了 Softmax，所以我们的模型不需要显式调用 Softmax
loss_function = nn.CrossEntropyLoss()

# 定义优化器
# 它会根据损失的梯度来更新模型的权重（也就是我们的词向量）
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练循环
epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for context, target in cbow_data:
        # 1. 准备数据：将数据转换为 PyTorch Tensors
        context_tensor = torch.tensor(context, dtype=torch.long)
        # 我们的模型输入是 [batch_size, context_size]，这里 batch_size=1
        context_tensor = context_tensor.view(1, -1) 
        
        target_tensor = torch.tensor([target], dtype=torch.long)

        # 2. 清零梯度
        # 在每次计算新的梯度之前，都需要将之前的梯度清零
        model.zero_grad()

        # 3. 前向传播
        # 将上下文传入模型，得到预测分数
        logits = model(context_tensor)

        # 4. 计算损失
        # 比较模型的预测和真实目标
        loss = loss_function(logits, target_tensor)

        # 5. 反向传播
        # 计算损失函数关于模型所有参数的梯度
        loss.backward()

        # 6. 更新权重
        # 优化器根据梯度更新参数（词向量）
        optimizer.step()

        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# 训练完成后，词向量就存在了里面
word_embeddings = model.embeddings.weight.data
print("\n训练完成！单词 'processes' 的词向量是:")
print(word_embeddings[word_to_ix['processes']])