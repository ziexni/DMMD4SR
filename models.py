import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, LayerNorm
from diffusion import SDNet, DiffusionProcess


class DMMD4SRModel(nn.Module):
    def __init__(self, args):
        super(DMMD4SRModel, self).__init__()

        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.n_clusters = args.n_clusters
        self.lambda_history = args.lambda_history
        self.lambda_intent = args.lambda_intent

        self.encoder = Encoder(args)

        # Multimodal
        self.text_embeddings = nn.Embedding(args.item_size, args.pretrain_emb_dim, padding_idx=0)
        self.img_embeddings = nn.Embedding(args.item_size, args.pretrain_emb_dim, padding_idx=0)

        # Projection layers
        self.fc_img = nn.Linear(args.pretrain_emb_dim, args.hidden_size)
        self.fc_text = nn.Linear(args.pretrain_emb_dim, args.hidden_size)

        # Diffusion model
        self.time_emb_dim = 16
        self.steps = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        in_dims = [args.hidden_size, args.hidden_size]
        out_dims = [args.hidden_size, args.hidden_size]

        # Domain shift denoising
        self.sdnet_td = SDNet(in_dims=in_dims, out_dims=out_dims, emb_size=self.time_emb_dim)
        self.sdnet_vd = SDNet(in_dims=in_dims, out_dims=out_dims, emb_size=self.time_emb_dim)
        
        # Interest-agnostic denoising
        self.sdnet_tc = SDNet(in_dims=in_dims, out_dims=out_dims, emb_size=self.time_emb_dim)
        self.sdnet_vc = SDNet(in_dims=in_dims, out_dims=out_dims, emb_size=self.time_emb_dim)
        self.sdnet_f = SDNet(in_dims=in_dims, out_dims=out_dims, emb_size=self.time_emb_dim)
        
        self.mu_text = nn.Linear(args.hidden_size, args.hidden_size)
        self.sigma_text = nn.Linear(args.hidden_size, args.hidden_size)
        self.mu_img = nn.Linear(args.hidden_size, args.hidden_size)
        self.sigma_img = nn.Linear(args.hidden_size, args.hidden_size)

        self.num_experts = args.num_experts
        self.text_experts = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in range(self.num_experts)])
        self.img_experts = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in range(self.num_experts)])

        self.fusion_layer = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LayerNorm(args.hidden_size),
            nn.ReLU()
        )

        self.gate = nn.Linear(args.hidden_size, self.num_experts)

        # ID projection
        self.id_projection = nn.Linear(args.hidden_size, args.hidden_size)
        self.mlp = nn.Sequential(nn.Linear(args.hidden_size * args.max_seq_length, self.n_clusters))

        # Condition fusion
        self.condition_fusion = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LayerNorm(args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size)
        )
        
        self.condition_gate = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.Sigmoid()
        )

        noise_schedule = "linear"
        noise_scale = [1, 0.1]
        noise_min = 0.0001
        noise_max = 0.02
        self.denoise_weight = 0.08
        self.diffusion_process = DiffusionProcess(
            noise_schedule=noise_schedule,
            noise_scale=noise_scale,
            noise_min=noise_min,
            noise_max=noise_max,
            steps=self.steps,
            device=self.device
        )

        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.apply(self.init_weights)

        # Load multimodal embeddings if available
        if args.is_use_mm:
            try:
                text_features_list = torch.load(self.args.text_embedding_path)
                self.text_embeddings.weight.data[1:-1, :] = text_features_list
                print("Loaded text embeddings")
            except:
                print("Text embeddings not found, using random initialization")
            
            if args.is_use_image:
                try:
                    image_features_list = torch.load(self.args.image_embedding_path)
                    self.img_embeddings.weight.data[1:-1, :] = image_features_list
                    print("Loaded image embeddings")
                except:
                    print("Image embeddings not found, using random initialization")

    def multimodal_fusion(self, text_embeddings, img_embeddings, item_embeddings=None):
        """Uncertainty-Guided Modality Denoising Fusion"""
        
        t_mu = self.mu_text(text_embeddings)
        t_sigma = torch.exp(self.sigma_text(text_embeddings))
        
        i_mu = self.mu_img(img_embeddings)
        i_sigma = torch.exp(self.sigma_img(img_embeddings))
        
        # Reparameterization
        t_z = t_mu + t_sigma * torch.randn_like(t_mu)
        i_z = i_mu + i_sigma * torch.randn_like(i_mu)
        
        t_gate = self.topk_routing_text(t_z)
        i_gate = self.topk_routing_img(i_z)
        
        # Process through experts
        expert_outputs = [expert(t_z) for expert in self.text_experts]
        stacked_expert_outputs = torch.stack(expert_outputs, dim=-2)
        gate_weights = t_gate.unsqueeze(-1)
        t_out = torch.sum(stacked_expert_outputs * gate_weights, dim=-2)

        expert_outputs = [expert(i_z) for expert in self.img_experts]
        stacked_expert_outputs = torch.stack(expert_outputs, dim=-2)
        gate_weights = i_gate.unsqueeze(-1)
        i_out = torch.sum(stacked_expert_outputs * gate_weights, dim=-2)
        
        fusion = item_embeddings + self.fusion_layer(torch.cat([t_out, i_out], dim=-1))
        
        return fusion, t_mu, t_sigma, i_mu, i_sigma

    def topk_routing_text(self, gate_logits, k=2):
        gate_logits = self.gate(gate_logits)
        routing_weights = F.softmax(gate_logits, dim=-1)
        top_k_weights, indices = torch.topk(routing_weights, k=k, dim=-1)
        routing_weights_new = torch.zeros_like(routing_weights)
        routing_weights_new.scatter_(-1, indices, top_k_weights)
        routing_weights_new = routing_weights_new / routing_weights_new.sum(dim=-1, keepdim=True)
        return routing_weights_new

    def topk_routing_img(self, gate_logits, k=2):
        gate_logits = self.gate(gate_logits)
        routing_weights = F.softmax(gate_logits, dim=-1)
        top_k_weights, indices = torch.topk(routing_weights, k=k, dim=-1)
        routing_weights_new = torch.zeros_like(routing_weights)
        routing_weights_new.scatter_(-1, indices, top_k_weights)
        routing_weights_new = routing_weights_new / routing_weights_new.sum(dim=-1, keepdim=True)
        return routing_weights_new
    
    def compute_kl_loss(self, mu, sigma):
        epsilon = 1e-8
        kl_loss = -0.5 * torch.mean(torch.sum(1 + torch.log(sigma.pow(2) + epsilon) - mu.pow(2) - sigma.pow(2), dim=-1))
        return kl_loss

    def get_modality_embeddings(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)

        text_embeddings = None
        img_embeddings = None
        
        if self.args.is_use_mm:
            text_embeddings = self.text_embeddings(sequence)
            text_embeddings = self.fc_text(text_embeddings)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
            
            img_embeddings = self.img_embeddings(sequence)
            img_embeddings = self.fc_img(img_embeddings)
            img_embeddings = F.normalize(img_embeddings, p=2, dim=-1)

        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb, text_embeddings, img_embeddings, item_embeddings
    
    def intent_cluster(self, input, n_clusters):
        X = input.view(input.shape[0], -1)
        centers = X[torch.randperm(X.size(0))[:n_clusters]].to(input.device)
        labels = self.mlp(X)
        labels = F.gumbel_softmax(labels, tau=0.1, hard=True)

        for i in range(self.n_clusters):
            if torch.sum(labels[:, i]) == 0:
                centers[i] = X[torch.randint(0, X.size(0), (1,))]
            else:
                centers[i] = torch.mean(X[labels[:, i].bool()], dim=0)
        return centers.view(n_clusters, input.shape[1], input.shape[-1]), torch.argmax(labels, dim=1)

    def forward(self, input_ids, is_train=False):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb, text_embeddings, img_embeddings, item_id_emb = self.get_modality_embeddings(input_ids)

        t_emb_loss = 0
        v_emb_loss = 0
        diff_loss = 0
        
        if self.args.is_use_mm and text_embeddings is not None and img_embeddings is not None:
            if is_train:
                # Domain shift denoising
                t_domain_terms = self.diffusion_process.caculate_losses(self.sdnet_td, text_embeddings)
                t_domain_loss = t_domain_terms['loss'].mean()
                
                v_domain_terms = self.diffusion_process.caculate_losses(self.sdnet_vd, img_embeddings)
                v_domain_loss = v_domain_terms['loss'].mean()
                
                text_embeddings_denoised = self.diffusion_process.p_sample(self.sdnet_td, text_embeddings, steps=5, sampling_noise=False)
                img_embeddings_denoised = self.diffusion_process.p_sample(self.sdnet_vd, img_embeddings, steps=5, sampling_noise=False)
                
                # Interest-agnostic denoising
                self.centroids, self.labels = self.intent_cluster(item_id_emb, self.n_clusters)
                id_condition = item_id_emb * self.lambda_history + self.centroids[self.labels] * self.lambda_intent
                
                # Text
                text_concat_features = torch.cat([text_embeddings_denoised, id_condition], dim=-1)
                text_gate = self.condition_gate(text_concat_features)
                text_fusion_features = self.condition_fusion(text_concat_features)
                text_conditioned_emb = text_embeddings_denoised * (1 - text_gate) + text_fusion_features * text_gate
                
                # Image
                img_concat_features = torch.cat([img_embeddings_denoised, id_condition], dim=-1)
                img_gate = self.condition_gate(img_concat_features)
                img_fusion_features = self.condition_fusion(img_concat_features)
                img_conditioned_emb = img_embeddings_denoised * (1 - img_gate) + img_fusion_features * img_gate
                
                t_condition_loss = self.diffusion_process.caculate_losses(self.sdnet_tc, text_conditioned_emb)['loss'].mean()
                v_condition_loss = self.diffusion_process.caculate_losses(self.sdnet_vc, img_conditioned_emb)['loss'].mean()
                
                text_embeddings_denoised = self.diffusion_process.p_sample(self.sdnet_tc, text_conditioned_emb, steps=5, sampling_noise=False)
                img_embeddings_denoised = self.diffusion_process.p_sample(self.sdnet_vc, img_conditioned_emb, steps=5, sampling_noise=False)
                
                enhanced_seq_emb, t_mu, t_sigma, i_mu, i_sigma = self.multimodal_fusion(text_embeddings_denoised, img_embeddings_denoised, sequence_emb)
                
                final_diff_loss = self.diffusion_process.caculate_losses(self.sdnet_f, enhanced_seq_emb)['loss'].mean()
                kl_loss = self.compute_kl_loss(t_mu, t_sigma) + self.compute_kl_loss(i_mu, i_sigma)

                t_emb_loss = t_domain_loss
                v_emb_loss = v_domain_loss
                diff_loss = 0.5 * (t_condition_loss + v_condition_loss) + 0.5 * final_diff_loss + 2 * kl_loss

            else:  # inference
                text_embeddings_denoised = self.diffusion_process.p_sample(self.sdnet_td, text_embeddings, steps=5, sampling_noise=False)
                img_embeddings_denoised = self.diffusion_process.p_sample(self.sdnet_vd, img_embeddings, steps=5, sampling_noise=False)
                
                self.centroids, self.labels = self.intent_cluster(item_id_emb, self.n_clusters)
                id_condition = item_id_emb * self.lambda_history + self.centroids[self.labels] * self.lambda_intent
                
                text_concat_features = torch.cat([text_embeddings_denoised, id_condition], dim=-1)
                text_gate = self.condition_gate(text_concat_features)
                text_fusion_features = self.condition_fusion(text_concat_features)
                text_conditioned_emb = text_embeddings_denoised * (1 - text_gate) + text_fusion_features * text_gate
                text_denoised = self.diffusion_process.p_sample(self.sdnet_tc, text_conditioned_emb, steps=5, sampling_noise=False)
                
                img_concat_features = torch.cat([img_embeddings_denoised, id_condition], dim=-1)
                img_gate = self.condition_gate(img_concat_features)
                img_fusion_features = self.condition_fusion(img_concat_features)
                img_conditioned_emb = img_embeddings_denoised * (1 - img_gate) + img_fusion_features * img_gate
                img_denoised = self.diffusion_process.p_sample(self.sdnet_vc, img_conditioned_emb, steps=5, sampling_noise=False)

                enhanced_seq_emb, _, _, _, _ = self.multimodal_fusion(text_embeddings_denoised, img_embeddings_denoised, sequence_emb)
                
                final_seq_emb = self.diffusion_process.p_sample(self.sdnet_f, enhanced_seq_emb, steps=5, sampling_noise=False)
             
                sequence_emb = (1 - self.denoise_weight) * sequence_emb + self.denoise_weight * final_seq_emb

        item_encoded_layers = self.encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
        sequence_output = item_encoded_layers[-1]
        
        return sequence_output, t_emb_loss, v_emb_loss, diff_loss

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
