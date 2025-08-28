import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        returns: x + positional encoding
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class QueryEncoder(nn.Module):
    """
    Query‐based Transformer Encoder (interleaved CLS tokens).
    - embed_dim: dimension of frame embeddings
    - n_heads, n_layers: TransformerEncoder config
    - window_size: interval at which to insert a CLS token
    """

    def __init__(self, embed_dim: int = 128, n_heads: int = 8, n_layers: int = 6, sliding_window_size: int = 32, dim_feedforward: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.sliding_window_size = sliding_window_size
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True, dim_feedforward=dim_feedforward)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (batch, T_frames, embed_dim)
        returns: h: (batch, T_cls, embed_dim)  -- the retrieved CLS tokens
        """
     
        # 1) Add positional encoding
        x = self.pos_encoder(frames)  # (B, T + n_cls, D)
        B, T, D = x.size()

        # 2) Build sliding‐window causal mask of shape (S, S)
        #    allow positions j where i - sliding_window_size < j <= i
        idxs = torch.arange(T, device=x.device)
        i = idxs[:, None]  # (S,1)
        j = idxs[None, :]  # (1,S)
        allowed = (j <= i) & ((i - j) < self.sliding_window_size)

        # PyTorch Transformer expects an additive mask: float with -inf at masked locations
        mask = torch.zeros((T, T), device=x.device, dtype=x.dtype)
        mask[~allowed] = float('-inf')

        # 3) Run through Transformer with our custom mask
        x = self.transformer(x, mask=mask)

        return x

class MAEDecoder(nn.Module):
    """
    Query‐based Transformer Encoder (interleaved CLS tokens).
    - embed_dim: dimension of frame embeddings
    - n_heads, n_layers: TransformerEncoder config
    - window_size: interval at which to insert a CLS token
    """

    """
    Query-based Transformer Encoder (interleaved CLS tokens).
    Keeps your sliding-window causal mask; uses RoPE instead of sinusoidal PE.
    """
    
    def __init__(self, embed_dim: int = 128, n_heads: int = 8, n_layers: int = 6,
                 sliding_window_size: int = 32, dim_feedforward: int = 512):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.sliding_window_size = sliding_window_size
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True, dim_feedforward=dim_feedforward)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim)

    def forward(self, x: torch.Tensor, masked_pos: torch.Tensor | None = None) -> torch.Tensor:
        """
        frames: (B, T, D)
        position_ids: optional (B, T) positions; if None, RoPE assumes 0..T-1 per sample.
        returns: (B, T, D)
        """
        x_pos_enc = self.pos_encoder(x)

        x = self.pos_encoder(x_pos_enc[:, masked_pos, :])  # (B, T + n_cls, D)

        x = self.transformer(x)  # (B, T + n_cls, D)

        return x

