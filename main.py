"""
main.py
ICLRec — 카테고리 제거, 텍스트/이미지만
베이스라인 조건: sampled 100-neg, leave-two-out, seed 없음, NDCG@10/HR@10/MRR
"""

import os
import argparse
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datamodule import DataModule
from models    import SASRecModel
from trainers  import ICLRecTrainer
from utils     import set_seed, check_path, load_item_features


def show_args(args):
    print("=" * 50)
    for k, v in vars(args).items():
        if not isinstance(v, np.ndarray):   # feat 배열은 출력 생략
            print(f"  {k:<30}: {v}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='ICLRec - MicroVideo (text+image only)')

    # ── 데이터 경로 ────────────────────────────────────────────────────
    parser.add_argument('--interaction_path', type=str, default='interaction.parquet')
    parser.add_argument('--item_path',        type=str, default='item_used.parquet')
    parser.add_argument('--title_npy',        type=str, default='title_emb.npy')

    # ── 출력 ──────────────────────────────────────────────────────────
    parser.add_argument('--output_dir',  type=str, default='output/')
    parser.add_argument('--data_name',   type=str, default='kuaishou')
    parser.add_argument('--model_idx',   type=int, default=0)

    # ── DataModule ────────────────────────────────────────────────────
    DataModule.add_to_argparse(parser)

    # ── 모델 하이퍼파라미터 ───────────────────────────────────────────
    parser.add_argument('--hidden_size',               type=int,   default=64)
    parser.add_argument('--num_hidden_layers',         type=int,   default=2)
    parser.add_argument('--num_attention_heads',       type=int,   default=2)
    parser.add_argument('--hidden_act',                type=str,   default='gelu')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.5)
    parser.add_argument('--hidden_dropout_prob',       type=float, default=0.5)
    parser.add_argument('--initializer_range',         type=float, default=0.02)
    parser.add_argument('--max_seq_length',            type=int,   default=50)

    # ── ICLRec 전용 ────────────────────────────────────────────────────
    parser.add_argument('--n_clusters',      type=int,   default=32)
    parser.add_argument('--lambda_history',  type=float, default=0.6)
    parser.add_argument('--lambda_intent',   type=float, default=0.4)
    parser.add_argument('--num_experts',     type=int,   default=4)
    parser.add_argument('--diff_loss',       type=float, default=0.01)
    parser.add_argument('--icl_loss',        type=float, default=0.001)
    parser.add_argument('--rec_weight',      type=float, default=1.0)

    # ── 멀티모달 (카테고리 없음) ──────────────────────────────────────
    parser.add_argument('--is_use_mm',       type=bool,  default=True)
    parser.add_argument('--pretrain_text_dim', type=int, default=512,
                        help='title_emb.npy 차원 (로드 후 자동 덮어씀)')
    parser.add_argument('--pretrain_img_dim',  type=int, default=512,
                        help='image_features.pt 차원 (로드 후 자동 덮어씀)')

    # ── 학습 ──────────────────────────────────────────────────────────
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--weight_decay',  type=float, default=0.0)
    parser.add_argument('--epochs',        type=int,   default=200)
    parser.add_argument('--early_stop_num',type=int,   default=10)
    parser.add_argument('--no_cuda',       action='store_true')
    parser.add_argument('--seed',          type=int,   default=2024)
    parser.add_argument('--gpu_id',        type=str,   default='0')
    parser.add_argument('--do_eval',       action='store_true')

    args = parser.parse_args()

    # ── 환경 설정 ──────────────────────────────────────────────────────
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    check_path(args.output_dir)

    # ── 로그 파일 ──────────────────────────────────────────────────────
    run_id      = f"ICLRec-{args.data_name}-{args.model_idx}"
    args.log_file = os.path.join(args.output_dir, run_id + '.txt')
    args.checkpoint_path = os.path.join(args.output_dir, run_id + '.pt')

    # ── DataModule ─────────────────────────────────────────────────────
    data = DataModule(args)
    data.setup()

    # ── 멀티모달 피처 로드 (텍스트 + 이미지, 카테고리 없음) ──────────
    text_feat, image_feat, text_dim, image_dim = load_item_features(
        item_path      = args.item_path,
        title_npy_path = args.title_npy,
    )
    args.pretrain_text_dim = text_dim
    args.pretrain_img_dim  = image_dim
    args.text_feat  = text_feat    # models.py에서 임베딩 초기화에 사용
    args.image_feat = image_feat

    # args 정리
    args.max_seq_length = args.max_len
    args.hidden_size    = args.hidden_size

    show_args(args)

    # ── 모델 ────────────────────────────────────────────────────────────
    model = SASRecModel(args)

    # ── DataLoader ──────────────────────────────────────────────────────
    train_loader   = data.train_dataloader()
    val_loader     = data.val_dataloader()
    test_loader    = data.test_dataloader()

    # cluster_dataloader = val_loader (ICLRecTrainer 인터페이스 유지)
    trainer = ICLRecTrainer(
        model, train_loader, val_loader,
        val_loader, test_loader, args
    )

    # ── 평가만 ─────────────────────────────────────────────────────────
    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print("Testing...")
        hit, ndcg, mrr = trainer.test(0)
        print(f"[Test] HR@10={hit:.4f}  NDCG@10={ndcg:.4f}  MRR={mrr:.4f}")
        return

    # ── 학습 루프 ──────────────────────────────────────────────────────
    best_val_ndcg  = 0.0
    best_test_hr   = best_test_ndcg = best_test_mrr = 0.0
    no_improve     = 0

    for epoch in range(1, args.epochs + 1):
        trainer.train(epoch)

        # valid
        val_hit, val_ndcg, val_mrr = trainer.valid(epoch)

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            no_improve    = 0
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"  ✓ Best valid NDCG@10={val_ndcg:.4f} — model saved")

            # 즉시 test
            test_hit, test_ndcg, test_mrr = trainer.test(epoch)
            best_test_hr   = test_hit
            best_test_ndcg = test_ndcg
            best_test_mrr  = test_mrr
        else:
            no_improve += 1
            print(f"  No improve: {no_improve}/{args.early_stop_num}")
            if no_improve >= args.early_stop_num:
                print("Early stopping.")
                break

    print("\n" + "=" * 50)
    print(f"Best Valid  NDCG@10 : {best_val_ndcg:.4f}")
    print(f"Best Test   HR@10   : {best_test_hr:.4f}")
    print(f"            NDCG@10 : {best_test_ndcg:.4f}")
    print(f"            MRR     : {best_test_mrr:.4f}")
    print("=" * 50)

    with open(args.log_file, 'a') as f:
        f.write(f"\nBest Test HR@10={best_test_hr:.4f} "
                f"NDCG@10={best_test_ndcg:.4f} MRR={best_test_mrr:.4f}\n")


if __name__ == '__main__':
    main()
