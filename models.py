"""
models.py
ICLRec SASRecModel — 카테고리 제거, 텍스트/이미지 피처만 사용
나머지 구조(diffusion, MoE, intent cluster)는 원본 유지
"""

import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import Encoder, LayerNorm
from diffusion import SDNet, DiffusionProcess


class SASRecModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # ── ID / position embedding ──────────────────────────────────────
        self.item_embeddings     = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        # ── 텍스트 / 이미지 (카테고리 제거) ─────────────────────────────
        self.text_embeddings = nn.Embedding(args.item_size, args.pretrain_text_dim, padding_idx=0)
        self.img_embeddings  = nn.Embedding(args.item_size, args.pretrain_img_dim,  padding_idx=0)

        self.fc_text = nn.Linear(args.pretrain_text_dim, args.hidden_size)
        self.fc_img  = nn.Linear(args.pretrain_img_dim,  args.hidden_size)

        # ── Transformer encoder ──────────────────────────────────────────
        self.encoder = Encoder(args)

        # ── Diffusion ────────────────────────────────────────────────────
        self.time_emb_dim = 16
        self.steps        = 32
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.denoise_weight = 0.08

        in_dims  = [args.hidden_size, args.hidden_size]
        out_dims = [args.hidden_size, args.hidden_size]

        self.sdnet_td = SDNet(in_dims, out_dims, self.time_emb_dim)
        self.sdnet_vd = SDNet(in_dims, out_dims, self.time_emb_dim)
        self.sdnet_tc = SDNet(in_dims, out_dims, self.time_emb_dim)
        self.sdnet_vc = SDNet(in_dims, out_dims, self.time_emb_dim)
        self.sdnet_f  = SDNet(in_dims, out_dims, self.time_emb_dim)

        self.diffusion_process = DiffusionProcess(
            noise_schedule="linear",
            noise_scale=[1, 0.1],
            noise_min=0.0001,
            noise_max=0.02,
            steps=self.steps,
            device=self.device,
        )

        # ── Uncertainty / MoE ────────────────────────────────────────────
        self.mu_text    = nn.Linear(args.hidden_size, args.hidden_size)
        self.sigma_text = nn.Linear(args.hidden_size, args.hidden_size)
        self.mu_img     = nn.Linear(args.hidden_size, args.hidden_size)
        self.sigma_img  = nn.Linear(args.hidden_size, args.hidden_size)

        self.num_experts  = args.num_experts
        self.text_experts = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size)
                                           for _ in range(self.num_experts)])
        self.img_experts  = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size)
                                           for _ in range(self.num_experts)])
        self.gate         = nn.Linear(args.hidden_size, self.num_experts)

        self.fusion_layer = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LayerNorm(args.hidden_size),
            nn.ReLU(),
        )

        # ── Intent cluster ───────────────────────────────────────────────
        self.n_clusters      = args.n_clusters
        self.lambda_history  = args.lambda_history
        self.lambda_intent   = args.lambda_intent
        self.id_projection   = nn.Linear(args.hidden_size, args.hidden_size)
        self.mlp             = nn.Sequential(
            nn.Linear(args.hidden_size * args.max_seq_length, self.n_clusters)
        )

        self.condition_fusion = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LayerNorm(args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
        )
        self.condition_gate = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.Sigmoid(),
        )

        # ── LayerNorm / dropout ──────────────────────────────────────────
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout   = nn.Dropout(args.hidden_dropout_prob)

        self.apply(self.init_weights)
        self._replace_mm_embeddings()

    # ------------------------------------------------------------------
    def _replace_mm_embeddings(self):
        """pretrained 피처로 임베딩 초기화 (텍스트/이미지)"""
        if hasattr(self.args, 'text_feat') and self.args.text_feat is not None:
            t = torch.FloatTensor(self.args.text_feat)   # (max_id+1, text_dim)
            self.text_embeddings.weight.data[:t.size(0)] = t

        if hasattr(self.args, 'image_feat') and self.args.image_feat is not None:
            v = torch.FloatTensor(self.args.image_feat)  # (max_id+1, img_dim)
            self.img_embeddings.weight.data[:v.size(0)]  = v

    # ------------------------------------------------------------------
    def get_modality_embeddings(self, sequence):
        seq_len      = sequence.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        item_emb = self.item_embeddings(sequence)
        pos_emb  = self.position_embeddings(position_ids)

        text_emb = self.fc_text(self.text_embeddings(sequence))
        text_emb = F.normalize(text_emb, p=2, dim=-1)

        img_emb  = self.fc_img(self.img_embeddings(sequence))
        img_emb  = F.normalize(img_emb, p=2, dim=-1)

        seq_emb  = self.LayerNorm(item_emb + pos_emb)
        seq_emb  = self.dropout(seq_emb)

        return seq_emb, text_emb, img_emb, item_emb

    # ------------------------------------------------------------------
    def _topk_routing(self, gate_logits, k=2):
        logits  = self.gate(gate_logits)
        weights = F.softmax(logits, dim=-1)
        top_w, idx = torch.topk(weights, k=k, dim=-1)
        new_w = torch.zeros_like(weights).scatter_(-1, idx, top_w)
        new_w = new_w / (new_w.sum(dim=-1, keepdim=True) + 1e-8)
        return new_w

    def multimodal_fusion(self, text_emb, img_emb, item_emb):
        t_mu    = self.mu_text(text_emb)
        t_sigma = torch.exp(self.sigma_text(text_emb))
        i_mu    = self.mu_img(img_emb)
        i_sigma = torch.exp(self.sigma_img(img_emb))

        t_z = t_mu + t_sigma * torch.randn_like(t_mu)
        i_z = i_mu + i_sigma * torch.randn_like(i_mu)

        t_gate = self._topk_routing(t_z)
        i_gate = self._topk_routing(i_z)

        t_experts = torch.stack([e(t_z) for e in self.text_experts], dim=-2)
        t_out = (t_experts * t_gate.unsqueeze(-1)).sum(dim=-2)

        i_experts = torch.stack([e(i_z) for e in self.img_experts], dim=-2)
        i_out = (i_experts * i_gate.unsqueeze(-1)).sum(dim=-2)

        fusion = item_emb + self.fusion_layer(torch.cat([t_out, i_out], dim=-1))
        return fusion, t_mu, t_sigma, i_mu, i_sigma

    def compute_kl_loss(self, mu, sigma):
        eps = 1e-8
        return -0.5 * torch.mean(
            torch.sum(1 + torch.log(sigma.pow(2) + eps) - mu.pow(2) - sigma.pow(2), dim=-1)
        )

    def intent_cluster(self, x, n_clusters):
        X      = x.view(x.size(0), -1)
        centers = X[torch.randperm(X.size(0))[:n_clusters]].to(x.device)
        labels  = F.gumbel_softmax(self.mlp(X), tau=0.1, hard=True)
        for i in range(n_clusters):
            if labels[:, i].sum() == 0:
                centers[i] = X[torch.randint(0, X.size(0), (1,))]
            else:
                centers[i] = X[labels[:, i].bool()].mean(dim=0)
        return centers.view(n_clusters, x.size(1), x.size(-1)), torch.argmax(labels, dim=1)

    # ------------------------------------------------------------------
    def forward(self, input_ids, is_train=False):
        attn_mask = (input_ids > 0).long()
        ext_mask  = attn_mask.unsqueeze(1).unsqueeze(2)
        max_len   = attn_mask.size(-1)
        sub_mask  = (torch.tril(torch.ones((1, max_len, max_len),
                                           device=input_ids.device)) == 0)
        ext_mask  = ext_mask * (~sub_mask).long()
        ext_mask  = ext_mask.to(dtype=next(self.parameters()).dtype)
        ext_mask  = (1.0 - ext_mask) * -10000.0

        seq_emb, text_emb, img_emb, item_id_emb = self.get_modality_embeddings(input_ids)

        t_emb_loss = v_emb_loss = diff_loss = 0.0

        if self.args.is_use_mm:
            if is_train:
                # Domain shift denoising
                t_domain_loss = self.diffusion_process.caculate_losses(self.sdnet_td, text_emb)['loss'].mean()
                v_domain_loss = self.diffusion_process.caculate_losses(self.sdnet_vd, img_emb)['loss'].mean()

                text_d = self.diffusion_process.p_sample(self.sdnet_td, text_emb, steps=5)
                img_d  = self.diffusion_process.p_sample(self.sdnet_vd, img_emb,  steps=5)

                # Interest-agnostic denoising
                self.centroids, self.labels = self.intent_cluster(item_id_emb, self.n_clusters)
                id_cond = item_id_emb * self.lambda_history + self.centroids[self.labels] * self.lambda_intent

                def cond_denoise(emb, cond):
                    cat   = torch.cat([emb, cond], dim=-1)
                    gate  = self.condition_gate(cat)
                    fused = self.condition_fusion(cat)
                    return emb * (1 - gate) + fused * gate

                text_c = cond_denoise(text_d, id_cond)
                img_c  = cond_denoise(img_d,  id_cond)

                t_cond_loss = self.diffusion_process.caculate_losses(self.sdnet_tc, text_c)['loss'].mean()
                v_cond_loss = self.diffusion_process.caculate_losses(self.sdnet_vc, img_c)['loss'].mean()

                text_c = self.diffusion_process.p_sample(self.sdnet_tc, text_c, steps=5)
                img_c  = self.diffusion_process.p_sample(self.sdnet_vc, img_c,  steps=5)

                enhanced, t_mu, t_sigma, i_mu, i_sigma = self.multimodal_fusion(text_c, img_c, seq_emb)

                final_diff_loss = self.diffusion_process.caculate_losses(self.sdnet_f, enhanced)['loss'].mean()
                kl_loss = self.compute_kl_loss(t_mu, t_sigma) + self.compute_kl_loss(i_mu, i_sigma)

                t_emb_loss = t_domain_loss
                v_emb_loss = v_domain_loss
                diff_loss  = 0.5 * (t_cond_loss + v_cond_loss) + 0.5 * final_diff_loss + 2 * kl_loss
                seq_emb    = enhanced

            else:
                text_d = self.diffusion_process.p_sample(self.sdnet_td, text_emb, steps=5)
                img_d  = self.diffusion_process.p_sample(self.sdnet_vd, img_emb,  steps=5)

                self.centroids, self.labels = self.intent_cluster(item_id_emb, self.n_clusters)
                id_cond = item_id_emb * self.lambda_history + self.centroids[self.labels] * self.lambda_intent

                def cond_denoise(emb, cond):
                    cat   = torch.cat([emb, cond], dim=-1)
                    gate  = self.condition_gate(cat)
                    fused = self.condition_fusion(cat)
                    return emb * (1 - gate) + fused * gate

                text_c = self.diffusion_process.p_sample(self.sdnet_tc, cond_denoise(text_d, id_cond), steps=5)
                img_c  = self.diffusion_process.p_sample(self.sdnet_vc, cond_denoise(img_d,  id_cond), steps=5)

                enhanced, _, _, _, _ = self.multimodal_fusion(text_c, img_c, seq_emb)
                final   = self.diffusion_process.p_sample(self.sdnet_f, enhanced, steps=5)
                seq_emb = (1 - self.denoise_weight) * seq_emb + self.denoise_weight * final

        item_encoded = self.encoder(seq_emb, ext_mask, output_all_encoded_layers=True)
        seq_out      = item_encoded[-1]

        return seq_out, t_emb_loss, v_emb_loss, diff_loss

    # ------------------------------------------------------------------
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, (LayerNorm, nn.LayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
