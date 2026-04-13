"""
datasets.py
ICLRec 기반 — 카테고리 제거, 텍스트/이미지 피처만 사용
평가 프로토콜: sampled 100-neg, leave-two-out, seed 없음
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
import copy


class RecDataset(Dataset):
    """
    학습/평가용 Dataset

    토큰 규칙:
        PAD    = 0
        item   = 1 ~ itemnum
        MASK   = itemnum + 1  (BERT4Rec 스타일, ICLRec 호환)

    mode:
        'train' : sliding-window, next-item prediction (SASRec 스타일)
        'valid' : train history → valid 정답 예측, sampled 100-neg
        'test'  : train+valid history → test 정답 예측, sampled 100-neg
    """

    def __init__(self, user_train, user_valid, user_test,
                 itemnum, maxlen,
                 neg_sample_size=100,
                 mode='train'):

        self.user_train    = user_train
        self.user_valid    = user_valid
        self.user_test     = user_test
        self.itemnum       = itemnum
        self.maxlen        = maxlen
        self.neg_sample_size = neg_sample_size
        self.mode          = mode

        if mode == 'train':
            # sequence 길이 >= 2 인 유저만 (context 1개 + target 1개 최소)
            self.users = [u for u, seq in user_train.items() if len(seq) >= 2]
        else:
            ref = user_valid if mode == 'valid' else user_test
            self.users = [u for u in user_train if ref.get(u)]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        if self.mode == 'train':
            return self._train_item(u)
        else:
            return self._eval_item(u)

    # ------------------------------------------------------------------
    def _train_item(self, u):
        """
        SASRec 스타일: seq[:-1] → seq[1:] 예측
        neg는 sequence-level (각 위치마다 1개)
        """
        seq    = self.user_train[u]
        input_ids  = seq[:-1]
        target_pos = seq[1:]
        seq_set    = set(seq)

        target_neg = [self._neg_sample(seq_set) for _ in input_ids]

        pad_len    = self.maxlen - len(input_ids)
        input_ids  = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids  = input_ids[-self.maxlen:]
        target_pos = target_pos[-self.maxlen:]
        target_neg = target_neg[-self.maxlen:]

        return (
            torch.LongTensor(input_ids),
            torch.LongTensor(target_pos),
            torch.LongTensor(target_neg),
        )

    def _eval_item(self, u):
        """
        평가: 정답 1개 + uniform random negative 100개
        rated = train 아이템만 제외 (valid 아이템 미포함 — BERT4Rec 베이스라인 동일)
        seed 고정 없음
        """
        train_seq = self.user_train[u]

        if self.mode == 'valid':
            history = train_seq
            target  = self.user_valid[u][0]
        else:
            history = train_seq + self.user_valid.get(u, [])
            target  = self.user_test[u][0]

        # 입력 시퀀스 구성
        item_seq  = history[-(self.maxlen - 1):]
        seq       = item_seq + [self.itemnum + 1]   # 마지막에 MASK 토큰
        pad_len   = self.maxlen - len(seq)
        seq       = [0] * pad_len + seq

        # negative sampling — train 아이템만 제외, seed 없음
        rated = set(train_seq)
        rated.add(0)
        negs = []
        while len(negs) < self.neg_sample_size:
            t = np.random.randint(1, self.itemnum + 1)
            if t not in rated:
                negs.append(t)

        candidates = [target] + negs          # 101개 (정답 첫 번째)
        labels     = [1] + [0] * self.neg_sample_size

        return (
            torch.LongTensor(seq),
            torch.LongTensor(candidates),
            torch.LongTensor(labels),
        )

    def _neg_sample(self, item_set):
        item = random.randint(1, self.itemnum)
        while item in item_set:
            item = random.randint(1, self.itemnum)
        return item
