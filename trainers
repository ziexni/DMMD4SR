import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import recall_at_k, ndcg_k, cal_mrr


class DMMD4SRTrainer:
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):
        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.model = model
        self.sim = self.args.sim

        if self.cuda_condition:
            self.model.cuda()

        self.train_dataloader = train_dataloader
        self.cluster_dataloader = cluster_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        return self.iteration(epoch, self.train_dataloader, train=True)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=True, train=True):
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            icl_avg_loss = 0.0
            joint_avg_loss = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, rec_batch in rec_data_iter:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                
                # Unpack batch
                if len(rec_batch) == 5:
                    user_ids, input_ids, target_pos_1, target_pos_2, answers = rec_batch
                    has_contrastive = True
                else:
                    user_ids, input_ids, target_pos, answers = rec_batch
                    has_contrastive = False

                # Forward
                sequence_output, t_emb_loss, v_emb_loss, diff_loss = self.model(input_ids, is_train=True)
                
                # Prediction loss
                if has_contrastive:
                    logits = self.predict_full(sequence_output[:, -1, :])
                    rec_loss = nn.CrossEntropyLoss()(logits, target_pos_1[:, -1])
                    
                    # Contrastive learning loss
                    icl_loss = self.cicl_loss((sequence_output, sequence_output), target_pos_1)
                else:
                    logits = self.predict_full(sequence_output[:, -1, :])
                    rec_loss = nn.CrossEntropyLoss()(logits, target_pos[:, -1])
                    icl_loss = 0

                # Total loss
                joint_loss = (self.args.rec_weight * rec_loss + 
                             self.args.icl_loss * (t_emb_loss + v_emb_loss) + 
                             self.args.diff_loss * diff_loss)
                
                if has_contrastive:
                    joint_loss += self.args.icl_loss * icl_loss

                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()
                if has_contrastive:
                    icl_avg_loss += icl_loss if isinstance(icl_loss, float) else icl_loss.item()
                joint_avg_loss += joint_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_data_iter)),
                "icl_avg_loss": "{:.4f}".format(icl_avg_loss / len(rec_data_iter)),
                "joint_avg_loss": "{:.4f}".format(joint_avg_loss / len(rec_data_iter)),
            }
            
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")

        else:
            # Evaluation
            self.model.eval()
            pred_list = None
            answer_list = None

            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, _, answers = batch[:4]

                with torch.no_grad():
                    sequence_output, _, _, _ = self.model(input_ids, is_train=False)
                    sequence_output = sequence_output[:, -1, :]
                    
                    # Full-sort ranking
                    rating_pred = self.predict_full(sequence_output)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    
                    # Top-20 for efficiency
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

            return self.get_full_sort_score(epoch, answer_list, pred_list)

    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def cicl_loss(self, coarse_intents, target_item):
        """Contrastive learning loss"""
        coarse_intent_1, coarse_intent_2 = coarse_intents[0], coarse_intents[1]
        sem_nce_logits, sem_nce_labels = self.info_nce(
            coarse_intent_1[:, -1, :], 
            coarse_intent_2[:, -1, :],
            self.args.temperature, 
            coarse_intent_1.shape[0], 
            self.sim,
            target_item[:, -1]
        )
        cicl_loss = nn.CrossEntropyLoss()(sem_nce_logits, sem_nce_labels)
        return cicl_loss

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot', intent_id=None):
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        
        if sim == 'cos':
            sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim_matrix = torch.mm(z, z.t()) / temp

        sim_i_j = torch.diag(sim_matrix, batch_size)
        sim_j_i = torch.diag(sim_matrix, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(batch_size)
        negative_samples = sim_matrix[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [1, 5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        mrr = cal_mrr(answers, pred_list)
        
        post_fix = {
            "Epoch": epoch,
            "HIT@1": "{:.4f}".format(recall[0]),
            "NDCG@1": "{:.4f}".format(ndcg[0]),
            "HIT@5": "{:.4f}".format(recall[1]),
            "NDCG@5": "{:.4f}".format(ndcg[1]),
            "HIT@10": "{:.4f}".format(recall[2]),
            "NDCG@10": "{:.4f}".format(ndcg[2]),
            "HIT@20": "{:.4f}".format(recall[4]),
            "NDCG@20": "{:.4f}".format(ndcg[4]),
            "MRR": "{:.4f}".format(mrr),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2], recall[3], ndcg[3], mrr], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
