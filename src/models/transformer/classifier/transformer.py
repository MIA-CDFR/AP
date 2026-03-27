import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        pad_idx: int = 0,
        seq_len: int = 512,
        d_model: int = 512,
        num_heads: int = 16,
        num_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.seq_len = seq_len
        self.n_classes = n_classes

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_idx,
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Attention-weighted pooling instead of mean pooling
        self.pooling_attn = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs
    ) -> SequenceClassifierOutput:
        if input_ids is None:
            raise ValueError("input_ids must be provided.")

        model_device = self.token_embedding.weight.device
        if input_ids.device != model_device:
            input_ids = input_ids.to(model_device)
        if attention_mask is not None and attention_mask.device != model_device:
            attention_mask = attention_mask.to(model_device)
        if labels is not None and labels.device != model_device:
            labels = labels.to(model_device)

        x = self.token_embedding(input_ids)  # [B, T, d_model]
        if x.size(1) > self.seq_len:
            raise ValueError(
                f"Input sequence length ({x.size(1)}) exceeds model seq_len ({self.seq_len}). "
                "Set tokenizer max_length <= model seq_len."
            )
        pos_embedding = self.pos_embedding[:, :x.size(1), :].to(x.device)
        x = x + pos_embedding

        if attention_mask is not None:
            pad_mask = attention_mask.eq(0)  # HF style: 1=token, 0=pad
        else:
            pad_mask = input_ids.eq(self.pad_idx)

        x = self.encoder(x, src_key_padding_mask=pad_mask)

        # Attention-weighted pooling: assign attention weights to each token
        attn_logits = self.pooling_attn(x)  # [B, T, 1]
        # Mask out padding tokens before softmax (use large negative number instead of -inf for MPS compatibility)
        attn_logits = attn_logits.masked_fill(pad_mask.unsqueeze(-1), -1e10)
        attn_weights = torch.softmax(attn_logits, dim=1)  # [B, T, 1]
        pooled = (x * attn_weights).sum(dim=1)  # [B, d_model]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            if self.n_classes == 1:
                loss = nn.MSELoss()(logits.squeeze(-1), labels.float())
            elif labels.dtype in (torch.long, torch.int64, torch.int32):
                # Label smoothing to prevent overconfidence
                # Use class weights if available for balanced training
                weight = getattr(self, 'class_weights', None)
                if weight is not None:
                    weight = weight.to(logits.device)  # Ensure weight is on same device as logits
                loss = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weight)(logits, labels.long())
            else:
                loss = nn.BCEWithLogitsLoss()(logits, labels.float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
