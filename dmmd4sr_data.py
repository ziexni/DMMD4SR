# dmmd4sr_data.py (기존 datasets.py 내용 그대로)
import random
import torch
import os
import pickle
from torch.utils.data import Dataset
from collections import defaultdict
import copy
import numpy as np


def DS(i_file, o_file, max_len):
    """
    Dynamic Segmentation operations
    """
    with open(i_file, "r+") as fr:
        data = fr.readlines()
    aug_d = {}
    max_save_len = max_len + 3
    max_keep_len = max_len + 2
    
    for d_ in data:
        u_i, item = d_.split(' ', 1)
        item = item.split(' ')
        item[-1] = str(eval(item[-1]))
        aug_d.setdefault(u_i, [])
        start = 0
        j = 3
        if len(item) > max_save_len:
            while start < len(item) - max_keep_len:
                j = start + 4
                while j < len(item):
                    if start < 1 and j - start < max_save_len:
                        aug_d[u_i].append(item[start:j])
                        j += 1
                    else:
                        aug_d[u_i].append(item[start:start + max_save_len])
                        break
                start += 1
        else:
            while j < len(item):
                aug_d[u_i].append(item[start:j + 1])
                j += 1
    
    with open(o_file, "w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i + " " + ' '.join(i_) + "\n")


class Generate_tag():
    def __init__(self, data_path, data_name, save_path):
        self.path = data_path
        self.data_name = data_name + "_1"
        self.save_path = save_path
    
    def generate(self):
        data_f = self.path + "/" + self.data_name + ".txt"
        train_dic = {}
        valid_dic = {}
        test_dic = {}
        
        if not os.path.exists(data_f):
            print(f"Warning: {data_f} not found. Skipping tag generation.")
            return
        
        with open(data_f, "r") as fr:
            data = fr.readlines()
            for d_ in data:
                items = d_.split(' ')
                tag_train = int(items[-3])
                tag_valid = int(items[-2])
                tag_test = int(items[-1])
                train_temp = list(map(int, items[:-3]))
                valid_temp = list(map(int, items[:-2]))
                test_temp = list(map(int, items[:-1]))
                if tag_train not in train_dic:
                    train_dic.setdefault(tag_train, [])
                train_dic[tag_train].append(train_temp)
                if tag_valid not in valid_dic:
                    valid_dic.setdefault(tag_valid, [])
                valid_dic[tag_valid].append(valid_temp)
                if tag_test not in test_dic:
                    test_dic.setdefault(tag_test, [])
                test_dic[tag_test].append(test_temp)

        total_dic = {"train": train_dic, "valid": valid_dic, "test": test_dic}
        print("Saving data to ", self.save_path)
        with open(self.save_path + "/" + self.data_name + "_t.pkl", "wb") as fw:
            pickle.dump(total_dic, fw)

    def load_dict(self, data_path):
        if not data_path:
            raise ValueError('invalid path')
        elif not os.path.exists(data_path):
            print("The dict not exist, generating...")
            self.generate()
        with open(data_path, 'rb') as read_file:
            data_dict = pickle.load(read_file)
        return data_dict

    def get_data(self, data_path, mode):
        data = self.load_dict(data_path)
        return data[mode]


class DMMD4SRDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

        # Create target item sets (for contrastive learning)
        if data_type == "train" and hasattr(args, 'train_data_file'):
            try:
                self.sem_tag = Generate_tag(self.args.data_dir, self.args.data_name, self.args.data_dir)
                self.train_tag = self.sem_tag.get_data(
                    self.args.data_dir + "/" + self.args.data_name + "_1_t.pkl", "train"
                )
            except Exception as e:
                print(f"Warning: Could not load contrastive tags: {e}")
                self.train_tag = None

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        copied_input_ids = copy.deepcopy(input_ids)
        pad_len = self.max_len - len(copied_input_ids)
        copied_input_ids = [0] * pad_len + copied_input_ids
        copied_input_ids = copied_input_ids[-self.max_len:]

        if isinstance(target_pos, tuple):
            pad_len_1 = self.max_len - len(target_pos[1])
            target_pos_1 = [0] * pad_len + target_pos[0]
            target_pos_2 = [0] * pad_len_1 + target_pos[1]
            target_pos_1 = target_pos_1[-self.max_len:]
            target_pos_2 = target_pos_2[-self.max_len:]
            assert len(target_pos_1) == self.max_len
            assert len(target_pos_2) == self.max_len
        else:
            target_pos = [0] * pad_len + target_pos
            target_pos = target_pos[-self.max_len:]
            assert len(target_pos) == self.max_len

        assert len(copied_input_ids) == self.max_len

        if isinstance(target_pos, tuple):
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos_1, dtype=torch.long),
                torch.tensor(target_pos_2, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )
        else:
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )
        return cur_rec_tensors

    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":
            input_ids = items[:-2]
            target_pos = items[1:-1]

            target_pos_ = target_pos
            target_pos = (target_pos, target_pos_)
            answer = [0]
        elif self.data_type == "valid":
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-2]]
        else:
            input_ids = items
            target_pos = items[1:]
            answer = [items[-1]]

        return self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)

    def __len__(self):
        return len(self.user_seq)
