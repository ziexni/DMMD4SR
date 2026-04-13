import os
import random
import numpy as np
import torch
from collections import defaultdict


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 분할 — leave-two-out
# ══════════════════════════════════════════════════════════════════════════════

def data_partition(fname):
    import pandas as pd

    df = pd.read_parquet(fname)
    df['user_id'] = df['user_id'] + 1
    df['item_id'] = df['item_id'] + 1
    df = df.sort_values(by=['user_id', 'timestamp'], kind='mergesort').reset_index(drop=True)

    usernum = int(df['user_id'].max())
    itemnum = int(df['item_id'].max())

    User = defaultdict(list)
    for u, i in zip(df['user_id'], df['item_id']):
        User[u].append(int(i))

    user_train, user_valid, user_test = {}, {}, {}
    for user, seq in User.items():
        n = len(seq)
        if n < 3:
            user_train[user] = seq
            user_valid[user] = []
            user_test[user]  = []
        else:
            user_train[user] = seq[:-2]
            user_valid[user] = [seq[-2]]
            user_test[user]  = [seq[-1]]

    print(f"[data_partition] users={usernum}, items={itemnum}")
    return [user_train, user_valid, user_test, usernum, itemnum]


# ══════════════════════════════════════════════════════════════════════════════
# 멀티모달 피처 로딩 — 텍스트 + 이미지(video_feature), 카테고리 제거
# ══════════════════════════════════════════════════════════════════════════════

def load_item_features(item_path, title_npy_path):
    import pandas as pd

    item_df   = pd.read_parquet(item_path)
    title_raw = np.load(title_npy_path)

    item_df = item_df.copy().reset_index(drop=True)
    item_df['item_id'] = item_df['item_id'] + 1
    max_item_id = int(item_df['item_id'].max())

    # 텍스트
    title_dim = title_raw.shape[1]
    text_feat = np.zeros((max_item_id + 1, title_dim), dtype=np.float32)
    for raw_idx, row in item_df.iterrows():
        iid = int(row['item_id'])
        if raw_idx < len(title_raw):
            text_feat[iid] = title_raw[raw_idx].astype(np.float32)

    # 이미지 (video_feature 컬럼)
    image_dim  = len(item_df['video_feature'].iloc[0])
    image_feat = np.zeros((max_item_id + 1, image_dim), dtype=np.float32)
    for _, row in item_df.iterrows():
        iid = int(row['item_id'])
        image_feat[iid] = np.array(row['video_feature'], dtype=np.float32)

    print(f"[load_item_features] title_dim={title_dim}, image_dim={image_dim}, "
          f"max_item_id={max_item_id}")
    return text_feat, image_feat, title_dim, image_dim


# ══════════════════════════════════════════════════════════════════════════════
# 평가 지표
# ══════════════════════════════════════════════════════════════════════════════

def get_metric(pred_list, topk=10):
    NDCG = HIT = MRR = 0.0
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT  += 1.0
    n = max(len(pred_list), 1)
    return HIT / n, NDCG / n, MRR / n


# ══════════════════════════════════════════════════════════════════════════════
# Early Stopping
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, checkpoint_path, patience=10, verbose=True):
        self.checkpoint_path = checkpoint_path
        self.patience   = patience
        self.verbose    = verbose
        self.counter    = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter    = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            if self.verbose:
                print(f"  ✓ Best model saved (NDCG@10={score:.4f})")
        else:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
