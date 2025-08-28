# pip install torchtune
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings  # RoPE (official PyTorch ecosystem)
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Optional



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
    
class RopeEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int = 128, n_heads: int = 8, dim_feedforward: int = 512, attn_dropout: float = 0.0):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # qkv + out proj
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # RoPE (caches freqs up to max_seq_len; will auto-expand if needed)
        self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=4096)

        # FFN
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, embed_dim),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn_dropout = attn_dropout

    def _shape_heads(self, x, B, S):
        # (B, S, D) -> (B, S, H, Hd)
        return x.view(B, S, self.n_heads, self.head_dim)

    def forward(self, x: torch.Tensor, 
                attn_mask: torch.Tensor | None = None, 
                position_ids: torch.Tensor | None = None,
                ):
        """
        x: (B, S, D), attn_mask: (S, S) additive mask with -inf where masked.
        position_ids: optional (B, S) if you're doing packed data; else None.
        """
        B, S, D = x.shape
        residual = x

        # --- MHA with RoPE ---
        x = self.norm1(x)
        q = self._shape_heads(self.q_proj(x), B, S)
        k = self._shape_heads(self.k_proj(x), B, S)
        v = self._shape_heads(self.v_proj(x), B, S)

        # apply RoPE to Q and K (expects [B, S, H, Hd])
        q = self.rope(q, input_pos=position_ids)
        k = self.rope(k, input_pos=position_ids)

        # fold heads for PyTorch SDPA: (B*H, S, Hd)
        q = q.permute(0, 2, 1, 3).reshape(B * self.n_heads, S, self.head_dim)
        k = k.permute(0, 2, 1, 3).reshape(B * self.n_heads, S, self.head_dim)
        v = v.permute(0, 2, 1, 3).reshape(B * self.n_heads, S, self.head_dim)

        # scaled dot-product attention with additive mask
        # attn_mask should be (S, S) -> broadcast to (B*H, S, S)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,           # bool mask: True=keep, False=mask
            dropout_p=(self.attn_dropout if self.training else 0.0),
            is_causal=False                # causality enforced by mask
        )
        attn_out = attn_out.reshape(B, self.n_heads, S, self.head_dim).permute(0, 2, 1, 3).reshape(B, S, D)
        x = residual + self.out_proj(attn_out)

        # --- FFN ---
        x = x + self.ff(self.norm2(x))
        return x


class QueryEncoder(nn.Module):
    """
    Query-based Transformer Encoder (interleaved CLS tokens).
    Keeps your sliding-window causal mask; uses RoPE instead of sinusoidal PE.
    """
    def __init__(self, embed_dim: int = 128, n_heads: int = 8, n_layers: int = 6,
                 sliding_window_size: int = 32, dim_feedforward: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.sliding_window_size = sliding_window_size
        self.layers = nn.ModuleList([
            RopeEncoderLayer(embed_dim, n_heads, dim_feedforward=dim_feedforward) for _ in range(n_layers)
        ])

    def forward(self, frames: torch.Tensor, position_ids: torch.Tensor | None = None, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        frames: (B, T, D)
        position_ids: optional (B, T) positions; if None, RoPE assumes 0..T-1 per sample.
        returns: (B, T, D)
        """
        x = frames
        B, T, D = x.size()

        # sliding-window causal mask: allow j <= i and (i - j) < window
        idxs = torch.arange(T, device=x.device)
        i = idxs[:, None]
        j = idxs[None, :]
        mask = (j <= i) & ((i - j) < self.sliding_window_size)

        for layer in self.layers:
            x = layer(x, attn_mask=mask, position_ids=position_ids)
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