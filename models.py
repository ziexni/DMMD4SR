import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import Encoder, LayerNorm
from diffusion import SDNet, DiffusionProcess


class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.args = args

        # ID / position
        self.item_embeddings     = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        # 텍스트 / 이미지 (카테고리 제거)
        self.text_embeddings = nn.Embedding(args.item_size, args.pretrain_text_dim, padding_idx=0)
        self.img_embeddings  = nn.Embedding(args.item_size, args.pretrain_img_dim,  padding_idx=0)
        self.fc_text = nn.Linear(args.pretrain_text_dim, args.hidden_size)
        self.fc_img  = nn.Linear(args.pretrain_img_dim,  args.hidden_size)

        # Transformer encoder
        self.encoder = Encoder(args)

        # Diffusion
        self.time_emb_dim   = 16
        self.steps          = 32
        self.denoise_weight = 0.08
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Uncertainty / MoE
        self.mu_text    = nn.Linear(args.hidden_size, args.hidden_size)
        self.sigma_text = nn.Linear(args.hidden_size, args.hidden_size)
        self.mu_img     = nn.Linear(args.hidden_size, args.hidden_size)
        self.sigma_img  = nn.Linear(args.hidden_size, args.hidden_size)

        self.num_experts  = args.num_experts
        self.text_experts = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size)
                                           for _ in range(self.num_experts)])
        self.img_experts  = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size)
                                           for _ in range(self.num_experts)])
        self.gate = nn.Linear(args.hidden_size, self.num_experts)

        self.fusion_layer = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LayerNorm(args.hidden_size),
            nn.ReLU(),
        )

        # Intent cluster
        self.n_clusters     = args.n_clusters
        self.lambda_history = args.lambda_history
        self.lambda_intent  = args.lambda_intent
        self.mlp = nn.Sequential(
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

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout   = nn.Dropout(args.hidden_dropout_prob)

        self.apply(self.init_weights)
        self.replace_embedding()

    def replace_embedding(self):
        if hasattr(self.args, 'text_feat') and self.args.text_feat is not None:
            t = torch.FloatTensor(self.args.text_feat)
            self.text_embeddings.weight.data[:t.size(0)] = t
        if hasattr(self.args, 'image_feat') and self.args.image_feat is not None:
            v = torch.FloatTensor(self.args.image_feat)
            self.img_embeddings.weight.data[:v.size(0)] = v

    def _topk_routing(self, x, k=2):
        weights = F.softmax(self.gate(x), dim=-1)
        top_w, idx = torch.topk(weights, k=k, dim=-1)
        new_w = torch.zeros_like(weights).scatter_(-1, idx, top_w)
        return new_w / (new_w.sum(dim=-1, keepdim=True) + 1e-8)

    def multimodal_fusion(self, text_emb, img_emb, item_emb):
        t_mu    = self.mu_text(text_emb)
        t_sigma = torch.exp(self.sigma_text(text_emb))
        i_mu    = self.mu_img(img_emb)
        i_sigma = torch.exp(self.sigma_img(img_emb))

        t_z = t_mu + t_sigma * torch.randn_like(t_mu)
        i_z = i_mu + i_sigma * torch.randn_like(i_mu)

        t_gate = self._topk_routing(t_z)
        i_gate = self._topk_routing(i_z)

        t_out = (torch.stack([e(t_z) for e in self.text_experts], dim=-2) * t_gate.unsqueeze(-1)).sum(dim=-2)
        i_out = (torch.stack([e(i_z) for e in self.img_experts],  dim=-2) * i_gate.unsqueeze(-1)).sum(dim=-2)

        fusion = item_emb + self.fusion_layer(torch.cat([t_out, i_out], dim=-1))
        return fusion, t_mu, t_sigma, i_mu, i_sigma

    def compute_kl_loss(self, mu, sigma):
        return -0.5 * torch.mean(
            torch.sum(1 + torch.log(sigma.pow(2) + 1e-8) - mu.pow(2) - sigma.pow(2), dim=-1)
        )

    def intent_cluster(self, x, n_clusters):
        X = x.view(x.size(0), -1)
        centers = X[torch.randperm(X.size(0))[:n_clusters]].to(x.device)
        labels  = F.gumbel_softmax(self.mlp(X), tau=0.1, hard=True)
        for i in range(n_clusters):
            mask = labels[:, i].bool()
            centers[i] = X[mask].mean(dim=0) if mask.sum() > 0 else X[torch.randint(0, X.size(0), (1,))]
        return centers.view(n_clusters, x.size(1), x.size(-1)), torch.argmax(labels, dim=1)

    def get_modality_embeddings(self, sequence):
        seq_len      = sequence.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        item_emb = self.item_embeddings(sequence)
        pos_emb  = self.position_embeddings(position_ids)
        text_emb = F.normalize(self.fc_text(self.text_embeddings(sequence)), p=2, dim=-1)
        img_emb  = F.normalize(self.fc_img(self.img_embeddings(sequence)),   p=2, dim=-1)

        seq_emb  = self.dropout(self.LayerNorm(item_emb + pos_emb))
        return seq_emb, text_emb, img_emb, item_emb

    def _cond_denoise(self, emb, cond):
        cat   = torch.cat([emb, cond], dim=-1)
        gate  = self.condition_gate(cat)
        fused = self.condition_fusion(cat)
        return emb * (1 - gate) + fused * gate

    def forward(self, input_ids, is_train=False):
        attn_mask = (input_ids > 0).long()
        ext_mask  = attn_mask.unsqueeze(1).unsqueeze(2)
        max_len   = attn_mask.size(-1)
        sub_mask  = torch.triu(torch.ones((1, max_len, max_len), device=input_ids.device), diagonal=1).bool()
        ext_mask  = ext_mask * (~sub_mask).long()
        ext_mask  = ext_mask.to(dtype=next(self.parameters()).dtype)
        ext_mask  = (1.0 - ext_mask) * -10000.0

        seq_emb, text_emb, img_emb, item_id_emb = self.get_modality_embeddings(input_ids)

        t_emb_loss = v_emb_loss = diff_loss = 0.0

        if self.args.is_use_mm:
            # Domain shift denoising
            text_d = self.diffusion_process.p_sample(self.sdnet_td, text_emb, steps=5)
            img_d  = self.diffusion_process.p_sample(self.sdnet_vd, img_emb,  steps=5)

            # Intent cluster condition
            self.centroids, self.labels = self.intent_cluster(item_id_emb, self.n_clusters)
            id_cond = (item_id_emb * self.lambda_history
                       + self.centroids[self.labels] * self.lambda_intent)

            # Interest-agnostic denoising
            text_c = self.diffusion_process.p_sample(self.sdnet_tc, self._cond_denoise(text_d, id_cond), steps=5)
            img_c  = self.diffusion_process.p_sample(self.sdnet_vc, self._cond_denoise(img_d,  id_cond), steps=5)

            enhanced, t_mu, t_sigma, i_mu, i_sigma = self.multimodal_fusion(text_c, img_c, seq_emb)

            if is_train:
                t_emb_loss = self.diffusion_process.caculate_losses(self.sdnet_td, text_emb)['loss'].mean()
                v_emb_loss = self.diffusion_process.caculate_losses(self.sdnet_vd, img_emb)['loss'].mean()
                t_cond_loss = self.diffusion_process.caculate_losses(self.sdnet_tc, self._cond_denoise(text_d, id_cond))['loss'].mean()
                v_cond_loss = self.diffusion_process.caculate_losses(self.sdnet_vc, self._cond_denoise(img_d,  id_cond))['loss'].mean()
                final_diff  = self.diffusion_process.caculate_losses(self.sdnet_f, enhanced)['loss'].mean()
                kl_loss     = self.compute_kl_loss(t_mu, t_sigma) + self.compute_kl_loss(i_mu, i_sigma)
                diff_loss   = 0.5 * (t_cond_loss + v_cond_loss) + 0.5 * final_diff + 2 * kl_loss
                seq_emb     = enhanced
            else:
                final   = self.diffusion_process.p_sample(self.sdnet_f, enhanced, steps=5)
                seq_emb = (1 - self.denoise_weight) * seq_emb + self.denoise_weight * final

        item_encoded = self.encoder(seq_emb, ext_mask, output_all_encoded_layers=True)
        return item_encoded[-1], t_emb_loss, v_emb_loss, diff_loss

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, (LayerNorm, nn.LayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
