"""
datamodule.py
카테고리 제거, 텍스트/이미지 피처만, leave-two-out split
"""

from torch.utils.data import DataLoader
from data import get_data
from datasets import RecDataset


INTERACTION_PATH = 'kuaishou_preprocess.pkl'


class DataModule:
    def __init__(self, args):
        self.args            = args
        self.max_len         = args.max_len
        self.neg_sample_size = args.neg_sample_size
        self.batch_size      = args.batch_size
        self.num_workers     = args.num_workers
        self.interaction_path = args.interaction_path

        self.user_train, self.user_valid, self.user_test, \
            self.usernum, self.itemnum = get_data(self.interaction_path)

        # args에 item_size 주입 (models.py에서 vocab_size로 사용)
        args.item_size = self.itemnum + 2   # 0: PAD, itemnum+1: MASK

    def setup(self):
        self.train_dataset = RecDataset(
            self.user_train, self.user_valid, self.user_test,
            self.itemnum, self.max_len,
            mode='train',
        )
        self.valid_dataset = RecDataset(
            self.user_train, self.user_valid, self.user_test,
            self.itemnum, self.max_len,
            neg_sample_size=self.neg_sample_size,
            mode='valid',
        )
        self.test_dataset = RecDataset(
            self.user_train, self.user_valid, self.user_test,
            self.itemnum, self.max_len,
            neg_sample_size=self.neg_sample_size,
            mode='test',
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument('--max_len',          type=int,   default=50)
        parser.add_argument('--neg_sample_size',  type=int,   default=100)
        parser.add_argument('--batch_size',       type=int,   default=256)
        parser.add_argument('--num_workers',      type=int,   default=4)
        parser.add_argument('--item_size',        type=int,   default=0)
        parser.add_argument('--interaction_path', type=str,   default=INTERACTION_PATH)
        return parser
