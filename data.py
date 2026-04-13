"""
data.py
leave-two-out split — BERT4Rec 베이스라인과 동일한 프로토콜
"""

import numpy as np
import pandas as pd
from collections import defaultdict


def get_data(interaction_path):
    """
    Returns:
        user_train, user_valid, user_test : {user_id: [item_id, ...]}
        usernum, itemnum
    """
    df = pd.read_parquet(interaction_path)

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
            # leave-two-out: 마지막 2개를 valid/test로
            user_train[user] = seq[:-2]
            user_valid[user] = [seq[-2]]
            user_test[user]  = [seq[-1]]

    print(f"[Split] users: {usernum}, items: {itemnum}")
    print(f"  train users with items: {sum(1 for v in user_train.values() if v)}")
    print(f"  valid users: {sum(1 for v in user_valid.values() if v)}")
    print(f"  test  users: {sum(1 for v in user_test.values() if v)}")

    return user_train, user_valid, user_test, usernum, itemnum
