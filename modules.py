import numpy as np
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias   = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    def __init__(self, args):
        super(Embeddings, self).__init__()
        self.item_embeddings     = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout   = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

    def forward(self, input_ids):
        seq_length   = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        embeddings   = self.item_embeddings(input_ids) + self.position_embeddings(position_ids)
        return self.dropout(self.LayerNorm(embeddings))


class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads)
            )
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size       = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key   = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense        = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm    = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout  = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_x_shape).permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        q = self.transpose_for_scores(self.query(input_tensor))
        k = self.transpose_for_scores(self.key(input_tensor))
        v = self.transpose_for_scores(self.value(input_tensor))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        scores = scores + attention_mask

        probs = nn.Softmax(dim=-1)(scores)
        probs = self.attn_dropout(probs)

        ctx = torch.matmul(probs, v)
        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(*ctx.size()[:-2], self.all_head_size)

        hidden = self.out_dropout(self.dense(ctx))
        return self.LayerNorm(hidden + input_tensor)


class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        self.intermediate_act_fn = ACT2FN[args.hidden_act] if isinstance(args.hidden_act, str) else args.hidden_act
        self.dense_2   = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout   = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden = self.intermediate_act_fn(self.dense_1(input_tensor))
        hidden = self.dropout(self.dense_2(hidden))
        return self.LayerNorm(hidden + input_tensor)


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention    = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        return self.intermediate(self.attention(hidden_states, attention_mask))


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
