import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from utils import get_metric


class Trainer:
    def __init__(self, model, train_dataloader, cluster_dataloader,
                 eval_dataloader, test_dataloader, args):

        self.args    = args
        self.cuda_condition = torch.cuda.is_available() and not args.no_cuda
        self.device  = torch.device("cuda" if self.cuda_condition else "cpu")
        self.model   = model.to(self.device)

        self.train_dataloader   = train_dataloader
        self.cluster_dataloader = cluster_dataloader
        self.eval_dataloader    = eval_dataloader
        self.test_dataloader    = test_dataloader

        self.optim = Adam(self.model.parameters(),
                          lr=self.args.lr,
                          weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum(p.nelement() for p in self.model.parameters()))

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch, dataloader, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list, split='valid'):
        hit, ndcg, mrr = get_metric(pred_list, topk=10)
        post_fix = {
            "Epoch": epoch,
            "HIT@10":  f"{hit:.4f}",
            "NDCG@10": f"{ndcg:.4f}",
            "MRR":     f"{mrr:.4f}",
        }
        print(f"[{split}]", post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(f"[{split}] {str(post_fix)}\n")
        return [hit, ndcg, mrr], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name, map_location=self.device))

    def predict_full(self, seq_out):
        """Full softmax over all items — DMMD4SR 원본과 동일"""
        test_item_emb = self.model.item_embeddings.weight  # (item_size, H)
        rating_pred   = torch.matmul(seq_out, test_item_emb.transpose(0, 1))  # (B, item_size)
        return rating_pred

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
        sim_mat = (F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
                   if sim == 'cos' else torch.mm(z, z.t()) / temp)

        positive = torch.cat((torch.diag(sim_mat, batch_size),
                               torch.diag(sim_mat, -batch_size)), dim=0).reshape(N, 1)
        mask     = self.mask_correlated_samples(batch_size)
        negative = sim_mat[mask].reshape(N, -1)

        labels = torch.zeros(N, dtype=torch.long, device=positive.device)
        logits = torch.cat((positive, negative), dim=1)
        return logits, labels


class ICLRecTrainer(Trainer):
    def __init__(self, model, train_dataloader, cluster_dataloader,
                 eval_dataloader, test_dataloader, args):
        super(ICLRecTrainer, self).__init__(
            model, train_dataloader, cluster_dataloader,
            eval_dataloader, test_dataloader, args
        )

    def iteration(self, epoch, dataloader, train=True):
        if train:
            self.model.train()
            rec_avg_loss = joint_avg_loss = 0.0
            data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, rec_batch in data_iter:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                # datasets.py train 반환: (user_id, input_ids, target_pos, answer)
                _, input_ids, target_pos, answer = rec_batch

                seq_out, t_emb_loss, v_emb_loss, diff_loss = self.model(input_ids, is_train=True)

                # ── CrossEntropy (full softmax) — DMMD4SR 원본과 동일 ──
                logits   = self.predict_full(seq_out)   # (B, item_size)
                rec_loss = nn.CrossEntropyLoss(ignore_index=0)(
                    logits.view(-1, logits.size(-1)),
                    target_pos.view(-1)
                )

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
            print(f"[Epoch {epoch}] rec_loss={rec_avg_loss/n:.4f} "
                  f"joint_loss={joint_avg_loss/n:.4f}")
            with open(self.args.log_file, "a") as f:
                f.write(f"[Epoch {epoch}] rec_loss={rec_avg_loss/n:.4f} "
                        f"joint_loss={joint_avg_loss/n:.4f}\n")

        else:
            # ── sampled 평가 (101-way) — 공정성 맞춘 평가 프로토콜 유지 ──
            self.model.eval()
            pred_list = []
            split     = 'valid' if dataloader is self.eval_dataloader else 'test'
            data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            with torch.no_grad():
                for i, batch in data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    seq, candidates, labels = batch   # datasets.py valid/test 반환 형식

                    seq_out, _, _, _ = self.model(seq, is_train=False)
                    seq_emb  = seq_out[:, -1, :]                            # (B, H)
                    cand_emb = self.model.item_embeddings(candidates)       # (B, 101, H)
                    scores   = torch.bmm(cand_emb,
                                         seq_emb.unsqueeze(-1)).squeeze(-1) # (B, 101)

                    for b in range(scores.size(0)):
                        rank = (scores[b][0] < scores[b]).sum().item() - 1
                        pred_list.append(rank)

            scores, result_info = self.get_sample_scores(epoch, pred_list, split=split)
            return scores, result_info
