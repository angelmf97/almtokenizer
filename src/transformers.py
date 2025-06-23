import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

#
# ──────────────────────────────────────────────────────────────────────────
#    1. POSITIONAL ENCODING (SINUSOIDAL)
# ──────────────────────────────────────────────────────────────────────────
#
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
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


#
# ──────────────────────────────────────────────────────────────────────────
#    2. QUERY‐BASED TRANSFORMER ENCODER
# ──────────────────────────────────────────────────────────────────────────
#
class QueryEncoder(nn.Module):
    """
    Query‐based Transformer Encoder (interleaved CLS tokens).
    - embed_dim: dimension of frame embeddings
    - n_heads, n_layers: TransformerEncoder config
    - window_size: interval at which to insert a CLS token
    """

    def __init__(self, embed_dim: int = 128, n_heads: int = 8, n_layers: int = 6, window_size: int = 6):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (batch, T_frames, embed_dim)
        returns: h: (batch, T_cls, embed_dim)  -- the retrieved CLS tokens
        """
     
        # 3) Add positional encoding
        x = self.pos_encoder(frames)  # (B, T + n_cls, D)

        # 4) Transformer expects (S, B, D)
        x = self.transformer(x)  # (T + n_cls, B, D)

        return x
    

#
# ──────────────────────────────────────────────────────────────────────────
#    4. QUERY‐BASED TRANSFORMER DECODER
# ──────────────────────────────────────────────────────────────────────────
#
class QueryDecoder(nn.Module):
    """
    Query‐based Transformer Decoder (interleaved mask tokens).
    - embed_dim: dimension for decoder embeddings
    - n_heads, n_layers: TransformerDecoder config
    - window_size: how many mask tokens per CLS token
    """

    def __init__(self, embed_dim: int = 512, n_heads: int = 8, n_layers: int = 6, window_size: int = 6):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        decoder_layer = TransformerDecoderLayer(d_model=embed_dim, nhead=n_heads)
        self.transformer = TransformerDecoder(decoder_layer, num_layers=n_layers)
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional encoding
        self.pos_decoder = PositionalEncoding(embed_dim)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        quantized_h: (batch, T_cls, embed_dim)
        returns: (batch, T_expanded, embed_dim)  -- to be reassembled and fed into UnPatchify
        """

        # 3) Positional encoding + TransformerDecoder
        x = self.pos_decoder(x)  # (B, T_expanded, D)

        out = self.transformer(x, target)  # (T_expanded, B, D)

        return out

