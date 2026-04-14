import random
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import neg_sample


class RecWithContrastiveLearningDataset(Dataset):
    """
    train  : (user_id, input_ids, target_pos, answer)  — CrossEntropy용 4-tuple
    valid  : (seq, candidates, labels)                  — 101-way sampled 평가
    test   : (seq, candidates, labels)                  — 101-way sampled 평가
    """

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args           = args
        self.user_seq       = user_seq
        self.test_neg_items = test_neg_items
        self.data_type      = data_type
        self.max_len        = args.max_seq_length

        self.user_train = args.user_train
        self.user_valid = args.user_valid
        self.user_test  = args.user_test
        self.itemnum    = args.item_size - 2   # PAD/MASK 제외 실제 item 수

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        assert self.data_type in {'train', 'valid', 'test'}

        items = self.user_seq[index]   # 해당 유저의 전체 시퀀스

        if self.data_type == 'train':
            # ── DMMD4SR 원본과 동일한 슬라이싱 ──
            input_ids  = items[:-3]
            target_pos = items[1:-2]
            answer     = [0]   # no use (CrossEntropy는 target_pos로 계산)

            pad_len    = self.max_len - len(input_ids)
            input_ids  = ([0] * pad_len + input_ids)[-self.max_len:]
            target_pos = ([0] * pad_len + target_pos)[-self.max_len:]

            return (
                torch.tensor(index,      dtype=torch.long),
                torch.tensor(input_ids,  dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(answer,     dtype=torch.long),
            )

        else:
            # ── valid / test : 101-way sampled 평가 ──
            u = index + 1   # user_train/valid/test는 1-based

            if self.data_type == 'valid':
                history  = self.user_train.get(u, [])
                val_list = self.user_valid.get(u, [])
                target   = val_list[0] if val_list else None
            else:
                history   = self.user_train.get(u, []) + self.user_valid.get(u, [])
                test_list = self.user_test.get(u, [])
                target    = test_list[0] if test_list else None

            if target is None:
                seq        = [0] * self.max_len
                candidates = [1] + [2] * 100
                labels     = [1]  + [0] * 100
                return (
                    torch.tensor(seq,        dtype=torch.long),
                    torch.tensor(candidates, dtype=torch.long),
                    torch.tensor(labels,     dtype=torch.long),
                )

            # 입력 시퀀스: [history] + [MASK]
            item_seq = history[-(self.max_len - 1):]
            seq      = item_seq + [self.args.mask_id]
            pad_len  = self.max_len - len(seq)
            seq      = [0] * pad_len + seq

            # 100 negatives — train 아이템만 제외
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
