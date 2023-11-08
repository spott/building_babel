"""
heavily inspired by llama (github.com/facebookresearch/llama)
and nanoGPT (github.com/karpathy/nanoGPT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.rope import RoPE

from .types import InitFunc_, TransformerConfig
from .modules.growable import GrowableEmbedding, GrowableLinear, GrowableRMSNorm
from typing import Optional, Callable

class Attention(nn.Module):
    def __init__(self, config: TransformerConfig, rope: RoPE):
        super().__init__()
        self.head_dim = config.head_dim
        self.dim = config.dim
        self.n_heads = self.dim // self.head_dim
        self.rope = rope
        assert self.n_heads * self.head_dim == self.dim

        # We phrase this in terms of the size of the heads and the size of the
        # embedding dimension, rather than the number of heads, as the number
        # of heads will change when we grow.

        # in and out weight transformations. We do all in transformations
        # (key, query, value) in a batch and split.
        self.wq_in = GrowableLinear(self.dim, self.dim, bias=False)
        self.wk_in = GrowableLinear(self.dim, self.dim, bias=False)
        self.wv_in = GrowableLinear(self.dim, self.dim, bias=False)
        self.w_out = GrowableLinear(self.dim, self.dim, bias=False)

    def grow(self, new_dim, init: InitFunc_ | None = None):
        assert new_dim % self.head_dim == 0
        self.wq_in.grow(new_dim, new_dim, init=init)
        self.wk_in.grow(new_dim, new_dim, init=init)
        self.wv_in.grow(new_dim, new_dim, init=init)
        self.w_out.grow(new_dim, new_dim, init=init)
        self.dim = new_dim

        self.n_heads = self.dim // self.head_dim
        assert self.n_heads * self.head_dim == self.dim

    def forward(self, x):
        bs, sl, _ = x.size()

        q = self.wq_in(x)
        k = self.wk_in(x)
        v = self.wv_in(x)

        q, k = self.rope.apply_rotary_emb(q, k)

        q = q.view(bs, sl, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, sl, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, sl, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bs, sl, self.dim)
        return self.w_out(y)


class FeedForwardNetwork(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.dim = config.dim
        self.mo = config.multiple_of
        self.hidden_dim = FeedForwardNetwork._round_to_multiple(
            4 * self.dim / 3, self.mo
        )

        self.w1 = GrowableLinear(self.dim, self.hidden_dim, bias=False)
        self.gate = GrowableLinear(self.dim, self.hidden_dim, bias=False)
        self.w2 = GrowableLinear(self.hidden_dim, self.dim, bias=False)

    @staticmethod
    def _round_to_multiple(hidden_dim: int | float, multiple: int) -> int:
        return int(multiple * ((hidden_dim + multiple - 1) // multiple))

    def grow(
        self,
        new_dim,
        init_w: InitFunc_ | None = None,
        init_gate: InitFunc_ | None = lambda x: x.zero_(),
    ):
        hidden_dim = 4 * new_dim / 3
        hidden_dim = FeedForwardNetwork._round_to_multiple(hidden_dim, self.mo)
        self.w1.grow(new_dim, hidden_dim, init=init_w)
        self.gate.grow(new_dim, hidden_dim, init=init_gate)
        self.w2.grow(hidden_dim, new_dim, init=init_w)
        self.dim = new_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.gate(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: TransformerConfig, rope: RoPE):
        super().__init__()
        self.dim = config.dim
        self.norm_eps = config.norm_eps
        self.layer_id = layer_id
        self.head_dim = config.head_dim
        self.multiple_of = config.multiple_of
        self.config = config
        self.attention = Attention(config, rope=rope)
        self.attention_norm = GrowableRMSNorm(self.dim, eps=self.norm_eps)
        self.feed_forward = FeedForwardNetwork(config)
        self.feed_forward_norm = GrowableRMSNorm(self.dim, eps=self.norm_eps)

    def grow(
        self,
        new_dim,
        init_attention: InitFunc_ | None = None,
        init_w: InitFunc_ | None = None,
        init_gate: InitFunc_ | None = None,
    ):
        self.dim = new_dim
        self.attention.grow(new_dim, init_attention)
        self.attention_norm.grow(new_dim)
        self.feed_forward.grow(new_dim, init_w, init_gate)
        self.feed_forward_norm.grow(new_dim)

    def forward(self, x: torch.Tensor):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.feed_forward_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.dim = config.dim
        self.rope = RoPE(self.dim, self.config.theta, self.config.max_seq_len)

        self.text_embeddings = GrowableEmbedding(self.vocab_size, self.dim)
        self.layers = nn.Sequential(
            *[TransformerBlock(i, config, self.rope) for i in range(self.n_layers)]
        )
        self.norm = GrowableRMSNorm(self.dim, eps=config.norm_eps)
        self.output = GrowableLinear(self.dim, self.vocab_size)

    def get_num_params(self, incl_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if not incl_embedding:
            n_params -= self.text_embeddings.weight.numel()
        return n_params

    def grow(self, new_dim: int | None = None, new_layers: int | None = None):
        assert new_dim or new_layers, "Must set one of new_dim or new_layers"
        if new_dim is not None:
            assert new_dim % self.config.multiple_of == 0

            for layer in self.layers:
                layer.grow(new_dim)
            self.norm.grow(new_dim)
            self.output.grow(new_dim, self.vocab_size)
            self.rope.grow(new_dim)
            self.text_embeddings.grow(new_dim)
            self.dim = new_dim

        if new_layers is not None:
            [
                self.layers.append(TransformerBlock(i, self.config, self.rope))
                for i in range(self.n_layers, new_layers)
            ]
            self.n_layers = new_layers

    def forward(self, tokens: torch.Tensor):
        b, t = tokens.size()
        x = self.text_embeddings(tokens)
        x = self.layers(x)
        x = self.norm(x)
        x = self.output(x).float()
        return x