import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

data_path= "/root/autodl-tmp/LLM4RecWithQwen/data"
embedding_path= os.path.join(data_path, "embeddings")
checkpoints_path = "/root/autodl-tmp/LLM4RecWithQwen/checkpoints"
os.makedirs(checkpoints_path, exist_ok=True)
os.makedirs(embedding_path, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -----------------------
# 模型定义（两层 MLP）（这个没啥好说的）
# -----------------------
class ItemTower(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.mlp(x)


class UserTower(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        # x: (batch, input_dim) or (batch, seq, input_dim) if you change
        # We assume x already pooled into (batch, input_dim)
        return self.mlp(x)


# -----------------------
# 数据读取与预处理
# -----------------------
def load_prepared_data():
    ratings_path = os.path.join(data_path, "ratings.csv")
    movies_path = os.path.join(data_path, "movies.csv")
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    return ratings, movies

def load_movie_init_embeddings():
    movie_ids = np.load(os.path.join(embedding_path, "movie_ids.npy"))
    movie_embeddings = np.load(os.path.join(embedding_path, "movie_embeddings.npy"))
    return movie_ids, movie_embeddings


# -----------------------
# 建立 user_history 和 user initial embedding （池化 word2vec）
# -----------------------
def build_user_history_and_init_embeddings(ratings, movie_ids, movie_embeddings):
    """
    返回：
      user_history: dict user_id -> [item_id,...]
      user_init_embs: dict user_id -> numpy array (emb_dim,)
    """
    # 转成 int (保证类型一致)
    ratings['user_id'] = ratings['user_id'].astype(int)
    ratings['item_id'] = ratings['item_id'].astype(int)
    
    #按用户聚合排序
    user_history = ratings.groupby("user_id")["item_id"].apply(list).to_dict()

    # map item id -> index in movie_ids
    movieId2index = {int(mid): idx for idx, mid in enumerate(movie_ids)}

    emb_dim = movie_embeddings.shape[1]
    user_init_emb = {}
    for uid, items in user_history.items():
        vecs = []
        for iid in items:
            if int(iid) in movieId2index:
                vecs.append(movie_embeddings[movieId2index[int(iid)]])
        if len(vecs) == 0:
            user_init_emb[uid] = np.zeros(emb_dim, dtype=np.float32)
        else:
            user_init_emb[uid] = np.mean(np.stack(vecs, axis=0), axis=0)
    return user_history, user_init_emb, movieId2index


# -----------------------
# BPR Dataset
# -----------------------
class BPRDataset(Dataset):
    def __init__(self, user_history, movie_index_list):
        """
        user_history: dict user_id -> [item_id,...]
        movie_index_list: list of all item_ids (actual ids) for negative sampling pool
        """
        self.users = list(user_history.keys())
        self.user_history = user_history
        self.all_items = movie_index_list

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_list = self.user_history[user]
        pos = random.choice(pos_list)
        # 负采样：从全量item抽取，直至不在pos_list
        neg = random.choice(self.all_items)
        while neg in pos_list:
            neg = random.choice(self.all_items)
        return int(user), int(pos), int(neg)


def collate_fn(batch):
    # batch: list of (user, pos, neg)
    users = torch.tensor([b[0] for b in batch], dtype=torch.long)
    pos = torch.tensor([b[1] for b in batch], dtype=torch.long)
    neg = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return users, pos, neg


# -----------------------
# BPR Loss
# -----------------------
def bpr_loss(u_vec, pos_vec, neg_vec, eps=1e-8):
    # u_vec, pos_vec, neg_vec: (batch, dim)
    pos_score = torch.sum(u_vec * pos_vec, dim=1)
    neg_score = torch.sum(u_vec * neg_vec, dim=1)
    loss = -torch.log(torch.sigmoid(pos_score - neg_score) + eps).mean()
    return loss


# -----------------------
# 训练函数
# -----------------------
def train(args):
    ratings, movies = load_prepared_data()
    movie_ids, movie_init_emb = load_movie_init_embeddings()
    user_history, user_init_emb_dict, movieId2index = build_user_history_and_init_embeddings(
        ratings, movie_ids, movie_init_emb
    )

    # create a list of all item ids for negative sampling
    all_item_ids = [int(x) for x in movie_ids.tolist()]

    # dataset & dataloader
    dataset = BPRDataset(user_history, all_item_ids)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    device = torch.device(DEVICE)

    emb_dim = movie_init_emb.shape[1]
    user_tower = UserTower(input_dim=emb_dim, hidden_dim=args.hidden_dim, output_dim=args.out_dim).to(device)
    item_tower = ItemTower(input_dim=emb_dim, hidden_dim=args.hidden_dim, output_dim=args.out_dim).to(device)

    optim = torch.optim.Adam(list(user_tower.parameters()) + list(item_tower.parameters()), lr=args.lr)

    # Pre-build tensors for fast lookup
    # map item_id -> index in movie_init_emb (movieId2index)
    # Also create arrays of item_init vectors in same order as all_item_ids
    item_init_vecs = movie_init_emb  # numpy (num_items, dim)
    item_id_to_pos = movieId2index

    # Precompute user initial pool embeddings tensor for fast lookup (numpy -> array aligned with user list)
    # We'll fetch per-batch by user id using the dict.
    print("[train] start training on device:", device)
    for epoch in range(args.epochs):
        user_tower.train(); item_tower.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_users, batch_pos, batch_neg in pbar:
            batch_users = batch_users.numpy().tolist()
            # prepare pos_init_vecs and neg_init_vecs and user_init_vecs
            pos_vecs = []
            neg_vecs = []
            user_vecs = []
            for u, pos_id, neg_id in zip(batch_users, batch_pos.tolist(), batch_neg.tolist()):
                # pos init vec
                pos_idx = item_id_to_pos.get(int(pos_id), None)
                neg_idx = item_id_to_pos.get(int(neg_id), None)
                if pos_idx is None or neg_idx is None:
                    # skip these samples (rare)
                    pos_vecs.append(np.zeros(emb_dim, dtype=np.float32))
                    neg_vecs.append(np.zeros(emb_dim, dtype=np.float32))
                else:
                    pos_vecs.append(item_init_vecs[pos_idx])
                    neg_vecs.append(item_init_vecs[neg_idx])

                # user init vec (pooling of user's history)
                user_init = user_init_emb_dict.get(int(u), np.zeros(emb_dim, dtype=np.float32))
                user_vecs.append(user_init)

            pos_vecs = torch.tensor(np.stack(pos_vecs, axis=0), dtype=torch.float32).to(device)
            neg_vecs = torch.tensor(np.stack(neg_vecs, axis=0), dtype=torch.float32).to(device)
            user_vecs = torch.tensor(np.stack(user_vecs, axis=0), dtype=torch.float32).to(device)

            optim.zero_grad()
            # forward through towers
            u_out = user_tower(user_vecs)       # (batch, out_dim)
            p_out = item_tower(pos_vecs)        # (batch, out_dim)
            n_out = item_tower(neg_vecs)        # (batch, out_dim)

            loss = bpr_loss(u_out, p_out, n_out)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n+1)})

        print(f"Epoch {epoch+1} finished. avg loss: {total_loss / max(1, len(loader)):.6f}")

        # 每轮保存临时 checkpoint
        torch.save(user_tower.state_dict(), os.path.join(checkpoints_path, "user_tower_epoch%d.pth" % (epoch+1)))
        torch.save(item_tower.state_dict(), os.path.join(checkpoints_path, "item_tower_epoch%d.pth" % (epoch+1)))

    # 训练完成，保存最终模型
    torch.save(user_tower.state_dict(), os.path.join(checkpoints_path, "user_tower.pth"))
    torch.save(item_tower.state_dict(), os.path.join(checkpoints_path, "item_tower.pth"))
    print("[train] saved checkpoints to", checkpoints_path)

    # 导出所有 item 的最终 embedding（一次性前向）
    item_tower.eval()
    with torch.no_grad():
        item_init_tensor = torch.tensor(item_init_vecs, dtype=torch.float32).to(device)
        transformed = item_tower(item_init_tensor).cpu().numpy()
    np.save(os.path.join(embedding_path, "item_tower_embeddings.npy"), transformed)
    np.save(os.path.join(embedding_path, "movie_ids.npy"), movie_ids)  # ensure backed up
    print("[train] saved item_tower_embeddings.npy")

    # 导出所有 user embedding（基于 user_init_emb_dict）
    user_ids = sorted(list(user_init_emb_dict.keys()))
    user_embs = []
    user_tower.eval()
    with torch.no_grad():
        for uid in tqdm(user_ids, desc="export user embeddings"):
            vec = user_init_emb_dict.get(uid, np.zeros(emb_dim, dtype=np.float32))
            vec_t = torch.tensor(vec.reshape(1, -1), dtype=torch.float32).to(device)
            out = user_tower(vec_t).cpu().numpy().reshape(-1)
            user_embs.append(out)
    np.save(os.path.join(embedding_path, "user_tower_embeddings.npy"), np.array(user_embs))
    np.save(os.path.join(embedding_path, "user_ids.npy"), np.array(user_ids))
    print("[train] saved user_tower_embeddings.npy and user_ids.npy")
    print("[train] done.")


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
