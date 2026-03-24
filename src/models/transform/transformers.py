import torch
from torch import nn

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, seq_len=32, d_model=128, num_heads=4):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.input_proj(x)

        # adicionar posição
        x = x + self.pos_embedding[:, :x.size(1), :]

        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        x = x.mean(dim=1)
        return self.classifier(x)