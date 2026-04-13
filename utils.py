"""
utils.py
베이스라인 조건에 맞춘 유틸리티
  - sampled 100-neg 평가
  - seed 고정 없음
  - 지표: NDCG@10, HR@10, MRR
"""

import os
import random
import numpy as np
import torch


# ══════════════════════════════════════════════════════════════════════════════
# 시드
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ══════════════════════════════════════════════════════════════════════════════
# 피처 로딩
# ══════════════════════════════════════════════════════════════════════════════

def load_item_features(item_path, title_npy_path):
    """
    텍스트(title_emb.npy)와 이미지(item_used.parquet의 video_feature 컬럼)만 로드.
    카테고리는 사용하지 않음.

    반환
    ----
    text_feat  : np.ndarray (max_item_id + 1, title_dim)
    image_feat : np.ndarray (max_item_id + 1, image_dim)
    title_dim, image_dim
    """
    import pandas as pd

    item_df   = pd.read_parquet(item_path)
    title_raw = np.load(title_npy_path)          # (num_items, title_dim)

    # item_id 1-based 변환 (data_partition과 동일)
    item_df = item_df.copy().reset_index(drop=True)
    item_df['item_id'] = item_df['item_id'] + 1
    max_item_id = int(item_df['item_id'].max())

    # 텍스트 피처
    title_dim  = title_raw.shape[1]
    text_feat  = np.zeros((max_item_id + 1, title_dim), dtype=np.float32)
    for raw_idx, row in item_df.iterrows():
        iid = int(row['item_id'])
        if raw_idx < len(title_raw):
            text_feat[iid] = title_raw[raw_idx].astype(np.float32)

    # 이미지 피처 — video_feature 컬럼에서 직접 로드
    sample_img = item_df['video_feature'].iloc[0]
    image_dim  = len(sample_img)
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
    """
    pred_list: 각 유저에 대한 정답 아이템의 rank (0-indexed)
    """
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
        self.patience  = patience
        self.verbose   = verbose
        self.counter   = 0
        self.best_ndcg = 0.0
        self.early_stop = False

    def __call__(self, ndcg, model):
        if ndcg > self.best_ndcg:
            self.best_ndcg = ndcg
            self.counter   = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            if self.verbose:
                print(f"  ✓ Best model saved (NDCG@10={ndcg:.4f})")
        else:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# ══════════════════════════════════════════════════════════════════════════════
# 기타
# ══════════════════════════════════════════════════════════════════════════════

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")
