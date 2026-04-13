import numpy as np
import math
import random
import os
import torch
from scipy.sparse import csr_matrix


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


class EarlyStopping:
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        if self.verbose:
            print(f"Validation score increased. Saving model...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def get_user_seqs(data_file):
    """
    Load user sequences from txt file
    
    Args:
        data_file: Path to txt file with format "user_id item1 item2 ..."
    
    Returns:
        user_seq: List of lists, each inner list is a user's item sequence
        max_item: Maximum item ID
        valid_rating_matrix: Sparse matrix for validation
        test_rating_matrix: Sparse matrix for test
    """
    lines = open(data_file).readlines()
    user_seq = []  # CRITICAL: Must be a list of lists
    item_set = set()
    
    for line in lines:
        user, items = line.strip().split(" ", 1)
        items = items.split(" ")
        items = [int(item) for item in items]
        user_seq.append(items)  # Append list to user_seq
        item_set = item_set | set(items)
    
    max_item = max(item_set)
    num_users = len(lines)
    num_items = max_item + 2
    
    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    
    return user_seq, max_item, valid_rating_matrix, test_rating_matrix


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    """Generate sparse rating matrix for validation (exclude last 2 items)"""
    row, col, data = [], [], []
    
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:
            row.append(user_id)
            col.append(item)
            data.append(1)
    
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))
    
    return rating_matrix


def generate_rating_matrix_test(user_seq, num_users, num_items):
    """Generate sparse rating matrix for test (exclude last 1 item)"""
    row, col, data = [], [], []
    
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:
            row.append(user_id)
            col.append(item)
            data.append(1)
    
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))
    
    return rating_matrix


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    true_users = 0
    
    for i in range(len(predicted)):
        act_set = set([actual[i]])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    
    return sum_recall / true_users


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, 1)
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] == actual[user_id]) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def cal_mrr(actual, predicted):
    sum_mrr = 0.
    for i in range(len(predicted)):
        r = []
        act_set = set([actual[i]])
        pred_list = predicted[i]
        for item in pred_list:
            if item in act_set:
                r.append(1)
            else:
                r.append(0)
        r = np.array(r)
        if np.sum(r) > 0:
            sum_mrr += np.reciprocal(np.where(r == 1)[0] + 1, dtype=np.float64)[0]
    return sum_mrr / len(predicted)
