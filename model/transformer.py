# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn.functional as F

from torch import nn
import math
from copy import deepcopy


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx

class RelativeEmbedding(nn.Module):
    def forward(self, input):
        '''
        :param input: [bsz x seqlen]
        :return: [max_len*2, embed_size]
        '''
        bsz, seq_len = input.size()[:2]
        max_pos = self.padding_idx + seq_len
        if max_pos > self.origin_shift:
            # recompute/expand embeddings if needed
            weights = self.get_embedding(
                max_pos * 2,
                self.embedding_dim,
                self.padding_idx,
            )
            weights = weights.to(self._float_tensor)
            del self.weights
            self.origin_shift = weights.size(0) // 2
            self.register_buffer('weights', weights)

        positions = torch.arange(-seq_len, seq_len).to(input.device).long() + self.origin_shift  # 2*seq_len

        # mask = input.eq(self.padding_idx)
        # positions.masked_fill_(mask, self.padding_idx)

        embed = self.weights.index_select(0, positions.long()).detach()
        return embed


class RelativeSinusoidalPositionalEmbedding(RelativeEmbedding):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        """

        :param embedding_dim:
        :param padding_idx:
        :param init_size:
        """
        super(RelativeSinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size % 2 == 0
        weights = self.get_embedding(
            init_size + 1,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('weights', weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        self.origin_shift = num_embeddings // 2 + 1
        return emb


class RelativePositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super(RelativePositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size % 2 == 0
        weights = self.get_embedding(
            init_size + 1,
            embedding_dim,
            padding_idx,
        )
        self.embed = nn.Parameter(weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        if hasattr(self, 'origin_shift'):
            raise RuntimeError("Cannot regenerate embedding")
        emb = nn.init.xavier_normal_(torch.randn(num_embeddings, embedding_dim))
        emb[padding_idx].fill_(0)
        self.origin_shift = num_embeddings // 2 + 1
        return emb

    def forward(self, input):
        bsz, seq_len = input.size()
        positions = torch.arange(-seq_len, seq_len).to(input.device).long() + self.origin_shift  # 2*seq_len

        embed = self.embed[positions.long()]
        return embed

class RelativeMultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout, r_w_bias=None, r_r_bias=None, scale=True,
                 padding_idx=0):
        """

        :param int d_model:
        :param int n_head:
        :param dropout: dropout on attention map
        :param r_w_bias: n_head x head_dim or None,
        :param r_r_bias: n_head x head_dim or None,
        :param scale:
        :param rel_pos_embed:
        """
        super(RelativeMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout_layer = nn.Dropout(dropout)

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.kv_linear = nn.Linear(d_model, d_model * 2, bias=False)
        self.r_linear = nn.Linear(self.head_dim, self.head_dim, bias=False)

        if scale:
            self.scale = math.sqrt(d_model // n_head)
        else:
            self.scale = 1

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
            self.r_w_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
        else:
            self.r_r_bias = r_r_bias  # r_r_bias = v
            self.r_w_bias = r_w_bias  # r_w_bias = u

    def forward(self, q, k, mask, pos_embed):
        """

        :param x: batch_size x max_len x d_model
        :param mask: batch_size x max_len x max_len
        :param weight: batch, max_len
        :return:
        """

        batch_size, max_len = q.size()[:2]

        r = self.r_linear(pos_embed)  # 2*max_len, d

        q = self.q_linear(q)  # batch_size x max_len x d_model
        kv = self.kv_linear(k)
        k, v = torch.chunk(kv, chunks=2, dim=-1)

        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)  # b x n_head x max_len x d_model

        rw_head_q = q + self.r_r_bias[:, None]
        AC = torch.einsum('bnqd,bnkd->bnqk', rw_head_q, k)  # b x n_head x max_len x d_model, n = head

        rr_head_q = q + self.r_w_bias[:, None]
        BD = torch.einsum('bnqd,ld->bnql', [rr_head_q, r])
        BD = self._shift(BD)
        attn = AC + BD

        attn = attn / self.scale  # batch, n_head, seq_len, seq_len

        attn = attn.masked_fill(mask[:, None, None, :].eq(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        v = torch.matmul(self.dropout_layer(attn), v).transpose(1, 2).reshape(batch_size, max_len, -1)

        return v

    def _shift(self, BD):
        """
        example:
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        to
        0   1  2
        -1  0  1
        -2 -1  0

        :param BD: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = BD.size()
        zero_pad = BD.new_zeros(bsz, n_head, max_len, 1)
        BD = torch.cat([BD, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)  # bsz x n_head x (2max_len+1) x max_len
        BD = BD[:, :, :-1].view(bsz, n_head, max_len, -1)  # bsz x n_head x 2max_len x max_len
        BD = BD[:, :, :, max_len:]
        return BD


class RelTransformerLayer(nn.Module):
    def __init__(self, d_model, self_attn, feedforward_dim, after_norm, dropout):
        super(RelTransformerLayer, self).__init__()

        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)

        self.self_attn = deepcopy(self_attn)

        self.after_norm = after_norm

        self.ffn = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(feedforward_dim, d_model),
                                 nn.Dropout(dropout),
                                 )

    def forward(self, h, mask, pos_embed):
        if not self.after_norm:
            h = self.norm1(h)

        residual_h = h

        attn_out_hh = self.self_attn(h, h, mask, pos_embed)  # batch, seq_len, d_model

        hh = attn_out_hh + residual_h
        if self.after_norm:
            hh = self.norm1(hh)

        if not self.after_norm:
            hh = self.norm2(hh)

        residual_hh = hh

        hh = self.ffn(hh)
        hh = residual_hh + hh
        if self.after_norm:
            hh = self.norm2(hh)

        return hh


class TwoStreamRelTransformerLayer(nn.Module):
    def __init__(self, d_model, self_attn, feedforward_dim, after_norm, dropout):
        super(TwoStreamRelTransformerLayer, self).__init__()

        self.h_norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.h_norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.self_attn_hh = deepcopy(self_attn)
        self.self_attn_hl = deepcopy(self_attn)

        self.after_norm = after_norm

        self.h_ffn = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(feedforward_dim, d_model),
                                   nn.Dropout(dropout),
                                   )

        self.l_ffn = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(feedforward_dim, d_model),
                                   nn.Dropout(dropout),
                                   )

    def forward(self, h, l, mask, pos_embed):
        if not self.after_norm:
            h = self.h_norm1(h)

        residual_h = h

        attn_out_hh = self.self_attn_hh(h, h, mask, pos_embed)  # batch, seq_len, d_model

        attn_out_hl = self.self_attn_hl(h, l, mask, pos_embed)  # batch, seq_len, d_model

        hh = attn_out_hh + residual_h
        if self.after_norm:
            hh = self.h_norm1(hh)

        if not self.after_norm:
            hh = self.h_norm2(hh)

        residual_hh = hh
        hh = self.h_ffn(hh)

        hh = residual_hh + hh
        if self.after_norm:
            hh = self.h_norm2(hh)
        hl = attn_out_hl
        return hh, hl


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, feedforward_dim, dropout, after_norm=True,
                 scale=True, dropout_attn=None, rel_pos_embed='sin', padding_idx=0):
        super(TransformerEncoder, self).__init__()
        if dropout_attn is None:
            dropout_attn = dropout

        if rel_pos_embed == 'sin':
            self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model // n_head, padding_idx, 512)
        elif rel_pos_embed == 'fix':
            self.pos_embed = RelativePositionalEmbedding(d_model // n_head, padding_idx)
        else:
            raise

        self_attn = RelativeMultiHeadAttn(d_model, n_head, dropout_attn, scale=scale, padding_idx=padding_idx)
        self.layers = nn.ModuleList(
            [RelTransformerLayer(d_model, self_attn, feedforward_dim, after_norm, dropout)
             for _ in range(num_layers-1)])
        self.two_stream = TwoStreamRelTransformerLayer(d_model, self_attn, feedforward_dim, after_norm, dropout)

    def forward(self, h, l, mask):
        pos_embed = self.pos_embed(mask)
        for layer in self.layers:
            h = layer(h, mask, pos_embed)
        hh, hl = self.two_stream(h, l, mask, pos_embed)
        return hh, hl
