import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):

        src = self.embedding(src)
        tgt = self.embedding(tgt)

        attn_output, _ = self.attention(tgt, src, src)

        x = self.norm1(tgt + attn_output)

        ffn_output = self.ffn(x)

        x = self.norm2(x + ffn_output)

        out = self.fc_out(x)

        return out