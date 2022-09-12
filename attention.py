from torch import Tensor, nn
from torch.nn.functional import softmax
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, k, heads=8):
        # k: dimension of queries, keys, values.
        # k * heads: dimension of embeddings (to be inputted to the model)
        super().__init__()
        self.d_model, self.heads = k, heads
        self.to_keys = nn.Linear(k, k * heads, bias=False)
        self.to_queries = nn.Linear(k, k * heads, bias=False)
        self.to_values = nn.Linear(k, k * heads, bias=False)
        self.unify_heads = nn.Linear(k * heads, k, bias=False)

    def forward(self, q, k, v):
        batch_size, seq_length, feature_dim = k.size()
        h = self.heads
        queries = self.to_queries(q).view(batch_size, seq_length, h, feature_dim)
        keys = self.to_keys(k).view(batch_size, seq_length, h, feature_dim)
        values = self.to_values(k).view(batch_size, seq_length, h, feature_dim)

        # Combine head dimension with batch dimension so that dot products can be computed with bmm
        queries = queries.transpose(1, 2).contiguous().view(batch_size * h, seq_length, feature_dim)
        keys = keys.transpose(1, 2).contiguous().view(batch_size * h, seq_length, feature_dim)
        values = values.transpose(1, 2).contiguous().view(batch_size * h, seq_length, feature_dim)

        # matrix dimensions: seq_length, feature_dim * feature_dim, seq_length
        # outputs: seq_length, seq_length
        # batch size: batch_size * h
        dot = torch.bmm(queries, keys.transpose(1, 2))

        # Perform softmax on each row
        dot = softmax(dot, dim=2)

        # matrix dimensions: seq_length, seq_length * seq_length, feature_dim
        # Then unpack batch_size * h dimension into 2 dimensions, producing original shape rank 4 tensor
        out = torch.bmm(dot, values).view(batch_size, h, seq_length, feature_dim)

        # Swap seq_length, h dimensions so that heads are together for each query
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, h, feature_dim)

        # Unify heads
        return self.unify_heads(out)


class SelfAttention(nn.Module):

    def __init__(self, k, heads=8):
        super().__init__()
        self.attention = MultiHeadAttention(k, heads=heads)

    def forward(self, x):
        return self.attention(x, x, x)
