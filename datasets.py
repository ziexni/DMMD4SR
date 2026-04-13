import random
import numpy as np
import torch
from torch.utils.data import Dataset
import copy

from utils import neg_sample


class RecWithContrastiveLearningDataset(Dataset):
    """
    학습/평가용 Dataset
    train : SASRec 스타일 next-item (sliding window)
    valid : train history → valid 정답, sampled 100-neg
    test  : train+valid history → test 정답, sampled 100-neg
    seed 고정 없음 — 베이스라인과 동일
    """

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args            = args
        self.user_seq        = user_seq   # list of item sequences (0-indexed user)
        self.test_neg_items  = test_neg_items
        self.data_type       = data_type
        self.max_len         = args.max_seq_length

        # user_train / user_valid / user_test 는 args에서 참조
        self.user_train = args.user_train
        self.user_valid = args.user_valid
        self.user_test  = args.user_test
        self.itemnum    = args.item_size - 2   # padding/mask 제외한 실제 item 수

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        assert self.data_type in {'train', 'valid', 'test'}

        items = self.user_seq[index]   # 해당 유저의 전체 시퀀스

        if self.data_type == 'train':
            input_ids  = items[:-3]
            target_pos = items[1:-2]
            answer     = [0]

            seq_set    = set(items)
            target_neg = [neg_sample(seq_set, self.args.item_size) for _ in input_ids]

            pad_len    = self.max_len - len(input_ids)
            input_ids  = [0] * pad_len + input_ids
            target_pos = [0] * pad_len + target_pos
            target_neg = [0] * pad_len + target_neg

            input_ids  = input_ids[-self.max_len:]
            target_pos = target_pos[-self.max_len:]
            target_neg = target_neg[-self.max_len:]

            return (
                torch.tensor(index,      dtype=torch.long),
                torch.tensor(input_ids,  dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer,     dtype=torch.long),
            )

        else:
            # valid / test
            # user_id는 user_seq의 index + 1 (1-based)
            u = index + 1

            if self.data_type == 'valid':
                history = self.user_train.get(u, [])
                target  = self.user_valid.get(u, [None])[0]
            else:
                history = self.user_train.get(u, []) + self.user_valid.get(u, [])
                target  = self.user_test.get(u, [None])[0]

            if target is None:
                # 정답 없는 유저 — 더미 반환
                seq        = [0] * self.max_len
                candidates = [1] + [2] * 100
                labels     = [1]  + [0] * 100
                return (
                    torch.tensor(seq,        dtype=torch.long),
                    torch.tensor(candidates, dtype=torch.long),
                    torch.tensor(labels,     dtype=torch.long),
                )

            # 입력 시퀀스
            item_seq = history[-(self.max_len - 1):]
            seq      = item_seq + [self.args.mask_id]
            pad_len  = self.max_len - len(seq)
            seq      = [0] * pad_len + seq

            # negative sampling — train 아이템만 제외, seed 없음
            rated = set(self.user_train.get(u, []))
            rated.add(0)
            negs = []
            while len(negs) < 100:
                t = np.random.randint(1, self.itemnum + 1)
                if t not in rated:
                    negs.append(t)

            candidates = [target] + negs
            labels     = [1] + [0] * 100

            return (
                torch.tensor(seq,        dtype=torch.long),
                torch.tensor(candidates, dtype=torch.long),
                torch.tensor(labels,     dtype=torch.long),
            )
