import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import RecWithContrastiveLearningDataset
from trainers import ICLRecTrainer
from models   import SASRecModel
from utils    import (EarlyStopping, set_seed, check_path,
                      data_partition, load_item_features)


def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        if not isinstance(getattr(args, arg), np.ndarray):
            print(f"{arg:<30} : {getattr(args, arg):>35}")


def main():
    parser = argparse.ArgumentParser()

    # 데이터 경로
    parser.add_argument("--data_dir",    default="./",           type=str)
    parser.add_argument("--output_dir",  default="output/",      type=str)
    parser.add_argument("--data_name",   default="kuaishou",     type=str)
    parser.add_argument("--interaction_path", default="interaction.parquet", type=str)
    parser.add_argument("--item_path",   default="item_used.parquet",        type=str)
    parser.add_argument("--title_npy",   default="title_emb.npy",            type=str)
    parser.add_argument("--model_idx",   default=0,              type=int)
    parser.add_argument("--do_eval",     action="store_true")
    parser.add_argument("--gpu_id",      default="0",            type=str)
    parser.add_argument("--no_cuda",     action="store_true")

    # 모델
    parser.add_argument("--hidden_size",                  default=64,    type=int)
    parser.add_argument("--num_hidden_layers",            default=2,     type=int)
    parser.add_argument("--num_attention_heads",          default=2,     type=int)
    parser.add_argument("--hidden_act",                   default="gelu",type=str)
    parser.add_argument("--attention_probs_dropout_prob", default=0.5,   type=float)
    parser.add_argument("--hidden_dropout_prob",          default=0.5,   type=float)
    parser.add_argument("--initializer_range",            default=0.02,  type=float)
    parser.add_argument("--max_seq_length",               default=50,    type=int)

    # ICLRec
    parser.add_argument("--n_clusters",     default=32,  type=int)
    parser.add_argument("--lambda_history", default=0.6, type=float)
    parser.add_argument("--lambda_intent",  default=0.4, type=float)
    parser.add_argument("--num_experts",    default=4,   type=int)
    parser.add_argument("--diff_loss",      default=0.01,type=float)
    parser.add_argument("--icl_loss",       default=0.001,type=float)
    parser.add_argument("--rec_weight",     default=1.0, type=float)
    parser.add_argument("--is_use_mm",      default=True,type=bool)

    # 멀티모달 차원 (로드 후 자동 설정)
    parser.add_argument("--pretrain_text_dim", default=512, type=int)
    parser.add_argument("--pretrain_img_dim",  default=512, type=int)

    # 학습
    parser.add_argument("--lr",             default=0.001, type=float)
    parser.add_argument("--batch_size",     default=256,   type=int)
    parser.add_argument("--epochs",         default=200,   type=int)
    parser.add_argument("--early_stop_num", default=10,    type=int)
    parser.add_argument("--weight_decay",   default=0.0,   type=float)
    parser.add_argument("--seed",           default=2024,  type=int)

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # ── 로그 / 체크포인트 경로 ──────────────────────────────────────────
    args_str             = f"ICLRec-{args.data_name}-{args.model_idx}"
    args.log_file        = os.path.join(args.output_dir, args_str + ".txt")
    args.checkpoint_path = os.path.join(args.output_dir, args_str + ".pt")

    # ── 데이터 분할 ─────────────────────────────────────────────────────
    dataset = data_partition(args.interaction_path)
    user_train, user_valid, user_test, usernum, itemnum = dataset

    args.user_train = user_train
    args.user_valid = user_valid
    args.user_test  = user_test
    args.item_size  = itemnum + 2   # 0: PAD, itemnum+1: MASK
    args.mask_id    = itemnum + 1

    # ── 멀티모달 피처 ───────────────────────────────────────────────────
    text_feat, image_feat, text_dim, image_dim = load_item_features(
        args.item_path, args.title_npy
    )
    args.pretrain_text_dim = text_dim
    args.pretrain_img_dim  = image_dim
    args.text_feat  = text_feat
    args.image_feat = image_feat

    show_args_info(args)
    with open(args.log_file, "a") as f:
        f.write(str(args) + "\n")

    # ── user_seq 구성 (datasets.py와 동일한 인터페이스) ─────────────────
    # user_seq[i] = user (i+1)의 전체 시퀀스 (train + valid + test)
    user_seq = []
    for u in range(1, usernum + 1):
        seq = user_train.get(u, []) + user_valid.get(u, []) + user_test.get(u, [])
        user_seq.append(seq)

    # ── Dataset / DataLoader ────────────────────────────────────────────
    train_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_loader  = DataLoader(train_dataset, sampler=train_sampler,
                               batch_size=args.batch_size)

    eval_dataset  = RecWithContrastiveLearningDataset(args, user_seq, data_type='valid')
    eval_sampler  = SequentialSampler(eval_dataset)
    eval_loader   = DataLoader(eval_dataset, sampler=eval_sampler,
                               batch_size=args.batch_size)

    test_dataset  = RecWithContrastiveLearningDataset(args, user_seq, data_type='test')
    test_sampler  = SequentialSampler(test_dataset)
    test_loader   = DataLoader(test_dataset, sampler=test_sampler,
                               batch_size=args.batch_size)

    # ── 모델 / 트레이너 ────────────────────────────────────────────────
    model   = SASRecModel(args=args)
    trainer = ICLRecTrainer(model, train_loader, eval_loader,
                            eval_loader, test_loader, args)

    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f"Load model from {args.checkpoint_path} for test!")
        scores, result_info = trainer.test(0)
        print(result_info)
        return

    # ── 학습 루프 ───────────────────────────────────────────────────────
    early_stopping = EarlyStopping(args.checkpoint_path,
                                   patience=args.early_stop_num, verbose=True)

    best_test_scores = None
    for epoch in range(1, args.epochs + 1):
        trainer.train(epoch)
        scores, _ = trainer.valid(epoch)

        # scores = [hit, ndcg, mrr], early stopping 기준 = NDCG@10
        early_stopping(scores[1], trainer.model)

        if early_stopping.counter == 0:
            # best 갱신 시점에 test
            test_scores, test_info = trainer.test(epoch)
            best_test_scores = test_scores
            print(f"  [Best Test] {test_info}")

        if early_stopping.early_stop:
            print("Early stopping.")
            break

    # ── 최종 결과 ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    if best_test_scores:
        print(f"Best Test  HR@10={best_test_scores[0]:.4f}  "
              f"NDCG@10={best_test_scores[1]:.4f}  "
              f"MRR={best_test_scores[2]:.4f}")
    print(f"{'='*50}")

    with open(args.log_file, "a") as f:
        if best_test_scores:
            f.write(f"\nBest Test HR@10={best_test_scores[0]:.4f} "
                    f"NDCG@10={best_test_scores[1]:.4f} "
                    f"MRR={best_test_scores[2]:.4f}\n")


if __name__ == "__main__":
    main()
