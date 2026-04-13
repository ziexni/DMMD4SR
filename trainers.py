"""
trainers.py
ICLRec Trainer — sampled 100-neg 평가, NDCG@10 / HR@10 / MRR
카테고리 제거, 텍스트/이미지만 사용
"""

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam

from utils import get_metric


class Trainer:
    def __init__(self, model, train_dataloader, cluster_dataloader,
                 eval_dataloader, test_dataloader, args):

        self.args    = args
        self.device  = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        self.model   = model.to(self.device)

        self.train_dataloader   = train_dataloader
        self.cluster_dataloader = cluster_dataloader
        self.eval_dataloader    = eval_dataloader
        self.test_dataloader    = test_dataloader

        self.optim = Adam(self.model.parameters(),
                          lr=self.args.lr,
                          weight_decay=self.args.weight_decay)

        print(f"Total Parameters: {sum(p.nelement() for p in self.model.parameters()):,}")

    # ------------------------------------------------------------------
    def train(self, epoch):
        return self.iteration(epoch, self.train_dataloader, train=True)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, train=False)

    # ------------------------------------------------------------------
    def _get_scores(self, pred_list):
        """pred_list: list of rank (0-indexed)"""
        hit, ndcg, mrr = get_metric(pred_list, topk=10)
        return hit, ndcg, mrr

    def _log(self, epoch, hit, ndcg, mrr, split='valid'):
        post_fix = {
            "epoch": epoch,
            f"HR@10":   f"{hit:.4f}",
            f"NDCG@10": f"{ndcg:.4f}",
            f"MRR":     f"{mrr:.4f}",
        }
        print(f"[{split}] {post_fix}")
        with open(self.args.log_file, "a") as f:
            f.write(f"[{split}] {str(post_fix)}\n")
        return hit, ndcg, mrr

    # ------------------------------------------------------------------
    def iteration(self, epoch, dataloader, train=True):
        raise NotImplementedError

    # ------------------------------------------------------------------
    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name, map_location=self.device))

    # ------------------------------------------------------------------
    # contrastive helpers
    def mask_correlated_samples(self, batch_size):
        N    = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if sim == 'cos':
            import torch.nn.functional as F
            sim_mat = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        else:
            sim_mat = torch.mm(z, z.t()) / temp

        sim_i_j      = torch.diag(sim_mat,  batch_size)
        sim_j_i      = torch.diag(sim_mat, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        mask = self.mask_correlated_samples(batch_size)
        negative_samples = sim_mat[mask].reshape(N, -1)

        labels  = torch.zeros(N, dtype=torch.long, device=positive_samples.device)
        logits  = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels


class ICLRecTrainer(Trainer):
    def __init__(self, model, train_dataloader, cluster_dataloader,
                 eval_dataloader, test_dataloader, args):
        super().__init__(model, train_dataloader, cluster_dataloader,
                         eval_dataloader, test_dataloader, args)

    def iteration(self, epoch, dataloader, train=True):
        if train:
            return self._train_epoch(epoch, dataloader)
        else:
            return self._eval_epoch(epoch, dataloader)

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch, dataloader):
        self.model.train()
        rec_avg_loss   = 0.0
        joint_avg_loss = 0.0

        data_iter = tqdm(enumerate(dataloader), total=len(dataloader),
                         desc=f"Epoch {epoch} [train]")

        for i, batch in data_iter:
            batch = tuple(t.to(self.device) for t in batch)
            # train batch: (input_ids, target_pos, target_neg)
            input_ids, target_pos, target_neg = batch

            seq_out, t_emb_loss, v_emb_loss, diff_loss = self.model(input_ids, is_train=True)

            # next-item prediction loss (마지막 위치)
            seq_emb    = seq_out[:, -1, :]                   # (B, H)
            pos_emb    = self.model.item_embeddings(target_pos[:, -1])  # (B, H)
            neg_emb    = self.model.item_embeddings(target_neg[:, -1])  # (B, H)

            pos_score  = (seq_emb * pos_emb).sum(dim=-1)    # (B,)
            neg_score  = (seq_emb * neg_emb).sum(dim=-1)    # (B,)

            rec_loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

            joint_loss = (self.args.rec_weight * rec_loss
                          + self.args.icl_loss  * (t_emb_loss + v_emb_loss)
                          + self.args.diff_loss * diff_loss)

            self.optim.zero_grad()
            joint_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optim.step()

            rec_avg_loss   += rec_loss.item()
            joint_avg_loss += joint_loss.item()

        n = max(len(dataloader), 1)
        print(f"[Epoch {epoch}] rec_loss={rec_avg_loss/n:.4f}  "
              f"joint_loss={joint_avg_loss/n:.4f}")
        with open(self.args.log_file, "a") as f:
            f.write(f"[Epoch {epoch}] rec_loss={rec_avg_loss/n:.4f} "
                    f"joint_loss={joint_avg_loss/n:.4f}\n")

    # ------------------------------------------------------------------
    def _eval_epoch(self, epoch, dataloader):
        """
        sampled 평가:
          - candidates[0] = 정답, candidates[1:] = negative 100개
          - rank = 정답 아이템이 101개 중 몇 위인지 (0-indexed)
          - seed 고정 없음 (Dataset에서 이미 샘플링됨)
        """
        self.model.eval()
        pred_list = []

        data_iter = tqdm(enumerate(dataloader), total=len(dataloader),
                         desc=f"Epoch {epoch} [eval]")

        with torch.no_grad():
            for i, batch in data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                seq, candidates, labels = batch

                seq_out, _, _, _ = self.model(seq, is_train=False)
                seq_emb = seq_out[:, -1, :]                  # (B, H)

                # candidate 임베딩: (B, 101, H)
                cand_emb = self.model.item_embeddings(candidates)

                # 점수: (B, 101)
                scores = torch.bmm(cand_emb, seq_emb.unsqueeze(-1)).squeeze(-1)

                # 정답(index 0)의 rank 계산 (내림차순)
                for b in range(scores.size(0)):
                    score_b = scores[b]           # (101,)
                    rank    = (score_b[0] <= score_b).sum().item() - 1
                    pred_list.append(rank)

        split = 'valid' if dataloader is self.eval_dataloader else 'test'
        hit, ndcg, mrr = self._log(epoch, *self._get_scores(pred_list), split=split)
        return hit, ndcg, mrr
