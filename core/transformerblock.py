from torch import nn
from core import SelfAttention


class TransformerBlock(nn.Module):

    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 3 * k),
            nn.ReLU(),
            nn.Linear(3 * k, k)
        )

    def forward(self, x):
        att = self.attention(x)
        y1 = self.norm1(att + x)
        return self.norm2(self.ff(y1) + y1)
