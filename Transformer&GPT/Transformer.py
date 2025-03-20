import numpy as np
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.norm_layer = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        residual = query

        # 线性变换
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 分割头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # 缩放点积注意力
        output, attn_weights = ScaledDotProductAttention()(Q, K, V, mask)

        # 合并头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 输出线性变换
        # output = self.norm_layer(output + residual)

        return output, attn_weights


# 测试代码
if __name__ == "__main__":
    d_model = 6
    num_heads = 2
    batch_size = 32
    seq_len = 10

    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, 1, seq_len)

    multihead_attn = MultiHeadAttention(d_model, num_heads)
    output, attn_weights = multihead_attn(query, key, value, mask)

    print("Output shape:", output.size())  