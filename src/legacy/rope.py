import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
import math
from ..patchify import ContinuousEncoder

# Rotary Position Embedding utilities
def rotary_embedding(q, k, seq_len, dim):
    # generate rotary positional embeddings
    theta = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=q.device).float() / dim))
    pos = torch.arange(seq_len, device=q.device).type_as(theta)
    freqs = torch.einsum('i,j->ij', pos, theta)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_emb = emb.cos()[None, :, None, :]
    sin_emb = emb.sin()[None, :, None, :]
    # apply to q,k shaped (seq_len, batch, heads, head_dim)
    q2 = (q * cos_emb) + (rotate_half(q) * sin_emb)
    k2 = (k * cos_emb) + (rotate_half(k) * sin_emb)
    return q2, k2

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class QueryEncoderRoPE(nn.Module):
    """
    Query Encoder with Rotary Positional Embedding and ContinuousEncoder backend.
    """
    def __init__(self, latent_dim=128, embed_dim=256, n_heads=8, n_layers=6, window_size=8):
        super().__init__()
        # self.continuous_encoder = ContinuousEncoder()  # use prior implementation
        self.proj = nn.Linear(latent_dim, embed_dim)
        self.unproj = nn.Linear(embed_dim, latent_dim)
        self.window_size = window_size
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, z):
        # 1) continuous encode
        # z is (B, latent_dim, T) (128, 128, 4)
        print("Z: ", z.shape)
        z = z.transpose(1, 2) # (B, T, latent_dim)
        print("Z after transpose: ", z.shape)
        x = self.proj(z)  # (B, T, embed_dim)
        x = z
        B, T, D = x.size()
        w = self.window_size

        print("Z before adjusting: ", z.shape)

        rem = T % w
        if rem != 0:
            pad_len = w - rem
            # pad zeros at end: pad format = (dim_end_left, dim_end_right)
            x = F.pad(x, (0,0,0,pad_len), mode='constant', value=0.0)  # (B, T+pad_len, D) :contentReference[oaicite:3]{index=3}
            T = T + pad_len
        
        B, T, D = x.size()
        
        print("Z after adjusting: ", z.shape)

        n_cls = T // w
        rem = T % w
        print(T, w, n_cls, rem)
        
        # prepare CLS tokens
        cls = self.cls_token.repeat(B, n_cls, 1)
        # interleave frames and cls tokens
        frames = x.view(B, n_cls, w, D)
        cls_ = cls.unsqueeze(2)
        seq = torch.cat([frames, cls_], dim=2).view(B, -1, D)  # (B, T + n_cls, D)
        # rotary embed on Q,K inside attention via hooks; here pass raw seq
        out = self.transformer(seq)
        # extract CLS positions
        pos = [(i+1)*(w+1)-1 for i in range(n_cls)]
        h = out[:, pos, :]
        return h
