import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i/d_model)))
                if i + 1 < d_model:
                    pe[pos, i+1] = math.cos(pos / (10000 ** (i/d_model)))

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SelfAttention(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        weights = torch.softmax(scores, dim=-1)

        output = torch.matmul(weights, V)

        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttention(d_model) for _ in range(num_heads)] )
        self.linear = nn.Linear(d_model * num_heads, d_model)

    def forward(self, x):

        head_outputs = [head(x) for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)
        output = self.linear(concat)

        return output


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff=256):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x):
        attn_output = self.attn(x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, vocab_size, d_model=64,
                 num_heads=4, d_ff=256, num_layers=2):

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff)
             for _ in range(num_layers)]
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerForLanguageModeling(nn.Module):

    def __init__(self, vocab_size, d_model=64,
                 num_heads=4, d_ff=256, num_layers=2):

        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)


class TransformerForTextClassification(nn.Module):

    def __init__(self, vocab_size, num_classes, d_model=64,
                 num_heads=4, d_ff=256, num_layers=2, dropout=0.1):

        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, attention_mask=None):
        encoded = self.encoder(x)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            summed = (encoded * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / lengths
        else:
            pooled = encoded.mean(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


Transformer = TransformerForLanguageModeling

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

vocab_size = 100
seq_len = 10
batch = 2
num_classes = 3

model = TransformerForTextClassification(vocab_size, num_classes=num_classes).to(device)

x = torch.randint(0, vocab_size, (batch, seq_len)).to(device)
attention_mask = torch.ones((batch, seq_len), dtype=torch.long).to(device)

y = model(x, attention_mask=attention_mask)

print(y.shape)
