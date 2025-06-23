import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from .rope import QueryEncoderRoPE
from ..patchify import ContinuousEncoder
from ..unpatchify import ContinuousDecoder



def interleave(visible: torch.Tensor,
               mask_tokens: torch.Tensor,
               keep_idx: torch.LongTensor,
               mask_idx: torch.LongTensor,
               seq_len: int = None) -> torch.Tensor:
    """
    Reconstruye la secuencia completa a partir de:
      • visible:    (B, T_vis, D)
      • mask_tokens:(B, T_mask, D)
      • keep_idx:   LongTensor de forma (T_vis,) con índices en [0..T_full)
      • mask_idx:   LongTensor de forma (T_mask,) con índices en [0..T_full)
      • seq_len:    longitud total T_full (opcional, se infiere si es None)

    Devuelve:
      full: (B, T_full, D) donde full[:, i, :] = visible[:, j, :] si i==keep_idx[j],
                                o mask_tokens[:, k, :] si i==mask_idx[k].
    """
    B, T_vis, D = visible.shape
    T_mask = mask_tokens.shape[1]
    if seq_len is None:
        seq_len = T_vis + T_mask

    # Creamos full y lo inicializamos a ceros (u otro valor si quisieras)
    full = visible.new_zeros((B, seq_len, D))

    # Preparamos los índices para scatter:
    # queremos un tensor (B, T_vis, D) → índices (B, T_vis, D) apuntando a la dimensión 1
    keep_idx_expanded = keep_idx.view(1, T_vis, 1).expand(B, -1, D)
    mask_idx_expanded = mask_idx.view(1, T_mask, 1).expand(B, -1, D)

    # scatter_ escribe visible y mask_tokens en sus posiciones
    full.scatter_(1, keep_idx_expanded, visible)
    full.scatter_(1, mask_idx_expanded, mask_tokens)

    return full
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
#    3. QUERY‐BASED TRANSFORMER ENCODER + RETRIEVAL
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
        frames = frames.transpose(1, 2)
        B, T, D = frames.size()
        w = self.window_size
        
        # Number of CLS tokens = T // w
        n_cls = T // w
        rem = T % w

        print("T:", T, "Number of CLS tokens:", n_cls, "Remaining frames:", rem)

        # 1) Expand CLS tokens:
        cls_tokens = self.cls_token.repeat(B, n_cls, 1)  # (B, n_cls, embed_dim)

        # 2) Interleave: split frames into (B, n_cls, w, D)
        frames_ = frames.view(B, n_cls, w, D)  # (B, n_cls, w, D)
        cls_tokens_ = cls_tokens.unsqueeze(2)  # (B, n_cls, 1, D)

        print("Frames shape:", frames_.shape, "\nCLS tokens shape:", cls_tokens_.shape)
        interleaved = torch.cat([cls_tokens_, frames_], dim=2)  # (B, n_cls, w+1, D)
        interleaved = interleaved.view(B, -1, D)  # (B, T + n_cls, D)

        # 3) Add positional encoding
        x = self.pos_encoder(interleaved)  # (B, T + n_cls, D)

        # 4) Transformer expects (S, B, D)
        x = self.transformer(x)  # (T + n_cls, B, D)

        # 5) Retrieve CLS tokens at every (w+1)‐th position
        #    positions = (w, 2w+1, 3w+2, ...)
        cls_positions = [(i + 1) * (w + 1) - 1 for i in range(n_cls)]
        h = x[:, cls_positions, :]  # (B, n_cls, embed_dim)
        return h


#
# ──────────────────────────────────────────────────────────────────────────
#    4. QUERY‐BASED TRANSFORMER DECODER + RETRIEVAL
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

    def forward(self, quantized_h: torch.Tensor) -> torch.Tensor:
        """
        quantized_h: (batch, T_cls, embed_dim)
        returns: (batch, T_expanded, embed_dim)  -- to be reassembled and fed into UnPatchify
        """
        B, T_cls, D = quantized_h.size()
        w = self.window_size

        # 1) Prepare mask tokens
        n_masks = T_cls * w
        mask_tokens = self.mask_token.repeat(B, n_masks, 1)  # (B, n_masks, D)

        # 2) Interleave: split quantized_h into (B, T_cls, 1, D), mask_tokens into (B, T_cls, w, D)
        quantized_h_ = quantized_h.view(B, T_cls, 1, D)
        mask_ = mask_tokens.view(B, T_cls, w, D)
        interleaved = torch.cat([mask_, quantized_h_], dim=2)  # (B, T_cls, w+1, D)
        interleaved = interleaved.view(B, -1, D)  # (B, T_expanded, D) where T_expanded = T_cls*(w+1)

        # 3) Positional encoding + TransformerDecoder
        x = self.pos_decoder(interleaved)  # (B, T_expanded, D)
        x = x.transpose(0, 1)  # (T_expanded, B, D)

        # We do not have a separate “memory” here; for simplicity we pass zeros
        memory = torch.zeros_like(x)
        out = self.transformer(x, memory)  # (T_expanded, B, D)
        out = out.transpose(0, 1)  # (B, T_expanded, D)

        return out


#
# ──────────────────────────────────────────────────────────────────────────
#    5. SEMANTIC RVQ (RESIDUAL VECTOR QUANTIZATION) WITH PRIORS
# ──────────────────────────────────────────────────────────────────────────
#
class SemanticRVQ(nn.Module):
    """
    Three‐layer RVQ with semantic priors.
    - num_layers: number of sequential VQ layers (paper uses 3)
    - codebook_size: e.g. 2048
    - embed_dim: dimensionality of code vectors (256 in paper)
    - semantic_prior: dict with 'speech' & 'sound' keys mapping to k-means centroids
    """

    def __init__(self, num_layers: int = 3, codebook_size: int = 2048, embed_dim: int = 256, semantic_prior=None):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            emb = nn.Embedding(codebook_size, embed_dim)
            if semantic_prior is not None:
                # Initialize half codebook with speech‐prior centroids, half with sound‐prior centroids
                emb.weight.data[: codebook_size // 2, :] = semantic_prior["speech"][: codebook_size // 2]
                emb.weight.data[codebook_size // 2 :, :] = semantic_prior["sound"][: codebook_size // 2]
            self.layers.append(emb)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, list]:
        """
        h: (batch, T_cls, embed_dim)
        Returns:
          - quantized_h: (batch, T_cls, embed_dim)
          - indices_per_layer: list of length=num_layers, each (batch, T_cls)
        """
        residual = h
        quantized_total = torch.zeros_like(h)
        all_indices = []

        for emb in self.layers:
            # Compute squared distances to each codebook vector
            # (batch, T_cls, codebook_size) = ||residual - emb.weight||^2
            dists = torch.sum((residual.unsqueeze(2) - emb.weight.unsqueeze(0).unsqueeze(0)) ** 2, dim=-1)
            indices = torch.argmin(dists, dim=-1)  # (batch, T_cls)
            all_indices.append(indices)
            quant = emb(indices)  # (batch, T_cls, embed_dim)
            quantized_total = quantized_total + quant
            residual = residual - quant

        return quantized_total, all_indices


#
# ──────────────────────────────────────────────────────────────────────────
#    6. CONTINUOUS AR TRANSFORMER (LATENT‐LEVEL “LM LOSS”)
# ──────────────────────────────────────────────────────────────────────────
#
class ContinuousARTransformer(nn.Module):
    """
    Lightweight continuous AR transformer for next‐step prediction in VQ latent space.
    - embed_dim: 256
    - n_heads, n_layers: configurable
    """

    def __init__(self, embed_dim: int = 256, n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(d_model=embed_dim, nhead=n_heads)
        self.transformer = TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.pos_enc = PositionalEncoding(embed_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (batch, T_cls, embed_dim)
        returns: (batch, T_cls, embed_dim)  -- predicted features for each position
        """
        B, T, D = features.size()
        x = self.pos_enc(features)  # (B, T, D)
        x = x.transpose(0, 1)  # (T, B, D)
        out = self.transformer(x, x)  # (T, B, D)
        return out.transpose(0, 1)  # (B, T, D)


#
# ──────────────────────────────────────────────────────────────────────────
#    7. MULTI‐SCALE DISCRIMINATOR (Hinge GAN)
# ──────────────────────────────────────────────────────────────────────────
#
class Discriminator(nn.Module):
    """
    Single‐scale discriminator (blocks of Conv1d + LeakyReLU).
    You would typically instantiate several with different hop lengths.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: list = [64, 128, 256, 512, 512, 512],
        kernel_sizes: list = [3, 3, 3, 3, 3, 3],
        strides: list = [1, 2, 2, 2, 2, 2],
    ):
        super().__init__()
        layers = []
        in_ch = in_channels
        for hd, ks, st in zip(hidden_dims, kernel_sizes, strides):
            layers.append(nn.Conv1d(in_ch, hd, kernel_size=ks, stride=st, padding=ks // 2))
            layers.append(nn.LeakyReLU(0.2))
            in_ch = hd
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, N_samples) or mel/log-mel spectrogram
        """
        return self.net(x)


#
# ──────────────────────────────────────────────────────────────────────────
#    8. FULL ALMTokenizer ASSEMBLY
# ──────────────────────────────────────────────────────────────────────────
#
class ALMTokenizer(nn.Module):
    """
    Combines:
      • Patchify (Encodec‐style),
      • QueryEncoder,
      • SemanticRVQ,
      • AR Transformer,
      • a linear projection into decoder dim,
      • QueryDecoder,
      • UnPatchify (Encodec‐style),
      • (and, optionally, MAE decoder for the first stage).

    Example usage (sketch):
        model = ALMTokenizer(patchify_args, encoder_args, decoder_args,
                              mae_args, vq_args, ar_args, unpatchify_args, window_size=6)
        x_hat = model(x)
        compute losses (reconstruction, adversarial, MAE, AR, etc.)
    """

    def __init__(
        self,
        encoder_args: dict,
        decoder_args: dict,
        mae_decoder_args: dict,
        vq_args: dict,
        ar_args: dict,
        window_size: int = 8,
    ):
        super().__init__()
        self.patchify = ContinuousEncoder()
        self.query_encoder = QueryEncoder(**encoder_args, window_size=window_size)
        self.vq = SemanticRVQ(**vq_args)
        # Project VQ's 256‐dim output into decoder embedding dim (512)
        self.decoder_proj = nn.Linear(vq_args["embed_dim"], decoder_args["embed_dim"])
        self.query_decoder = QueryDecoder(**decoder_args, window_size=window_size)
        self.unpatchify = ContinuousDecoder()
        self.pos_encoder = PositionalEncoding(encoder_args["embed_dim"])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_args["embed_dim"]))
        # MAE decoder (to reconstruct masked frames in stage 1)
        self.mae_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=encoder_args["embed_dim"], nhead=encoder_args["n_heads"]),
            num_layers=mae_decoder_args["n_layers"],
        )
        # AR transformer for latent next‐step prediction
        self.ar_model = ContinuousARTransformer(**ar_args)

    def forward(
        self, x: torch.Tensor, mask_rate: float = 0.25, w: int = None
    ) -> torch.Tensor:
        """
        x: (batch, 1, N_samples)
        mask_rate: for MAE masking in stage1
        w: optional override for window_size
        returns: reconstructed waveform x_hat: (batch, 1, N_samples)
        """
        # ===== STAGE 1: MAE PRETRAINING (optional sketch) =====
        e = self.patchify.encode(x)  # (B, T_frames, encoder_dim)
        print("Encoded frames shape:", e.shape)
        # Sketch of MAE loss: mask some frames, pass through encoder+mae_decoder to reconstruct
        B, T, D = e.shape
        num_mask = int(T * mask_rate)
        mask_indices = torch.randperm(T)[:num_mask].to(x.device)
        e_masked = e.clone()
        e_masked[:, mask_indices, :] = 0
        # Positional encode
        e_pos = PositionalEncoding(D)(e_masked)  # you could add a PositionalEncoding here if desired
        # MAE decoding (simply predict only masked positions; for brevity, not fully detailed)
        mae_out = self.mae_decoder(e_pos.transpose(0,1), e_pos.transpose(0,1))
        mae_out = mae_out.transpose(0,1)  # shape (B, T, D)
        mae_loss = F.mse_loss(mae_out[:, mask_indices, :], e[:, mask_indices, :])

        # ===== STAGE 2: FULL CODEC TRAINING =====
        # 1) QueryEncoder (possibly override w if provided)
        if w is not None:
            self.query_encoder.window_size = w
            self.query_decoder.window_size = w

        h = self.query_encoder(e)  # (B, T_cls, encoder_dim)

        # 2) Semantic RVQ
        quantized_h, q_indices = self.vq(h)  # quantized_h: (B, T_cls, encoder_dim)

        # 3) AR latent prediction (for AR loss)
        ar_pred = self.ar_model(quantized_h)  # (B, T_cls, encoder_dim)
        # ar_loss = F.mse_loss(ar_pred, quantized_h.detach())

        # 4) Project into decoder dimension
        dec_in = self.decoder_proj(quantized_h)  # (B, T_cls, decoder_dim)

        # 5) QueryDecoder => e_hat_frames: (B, T_expanded, decoder_dim)
        dec_out = self.query_decoder(dec_in)

        # For simplicity, assume dec_out aligns framewise with e (in actual code you'd retrieve appropriately)
        e_hat = dec_out  # (B, T_frames, encoder_dim) after some linear or reshape
        print("E_hat shape:", e_hat.shape)

        # 6) UnPatchify => waveform reconstruction
        x_hat = self.unpatchify.decode(e_hat)  # (B, 1, N_samples_reconstructed)

        return x_hat


    def forward_mae(
        self, x: torch.Tensor, mask_rate: float = 0.5, w: int = None
    ) -> torch.Tensor:
        """
        x: (batch, 1, N_samples)
        mask_rate: for MAE masking in stage1
        w: optional override for window_size
        returns: reconstructed waveform x_hat: (batch, 1, N_samples)
        """
        # ===== STAGE 1: MAE PRETRAINING (optional sketch) =====
        e = self.patchify.encode(x)  # (B, T_frames, encoder_dim)
        # Sketch of MAE loss: mask some frames, pass through encoder+mae_decoder to reconstruct
        B, D, T = e.shape
        
        # 2) Masking
        num_mask = int(T * mask_rate)
        perm = torch.randperm(T, device=x.device)
        mask_idx = perm[:num_mask]
        keep_idx = perm[num_mask:]

        # 3) Encoder on visible tokens
        e_vis = e[:, :, keep_idx]
        print("Visible frames shape:", e_vis.shape)  # (B, T_vis, D)
        h_enc = self.query_encoder(e_vis)
        print("Encoded visible frames shape:", h_enc.shape)

        mask_tokens = self.mask_token.repeat(B, num_mask, 1)
        dec_in = interleave(h_enc, mask_tokens, keep_idx, mask_idx)
        print("Decoder input shape:", dec_in.shape)  # (B, T_expanded, embed_dim)
        
        # Positional encode
        dec_in = self.pos_encoder(dec_in)  # (B, T_expanded, embed_dim)
        dec_out = self.mae_decoder(
                dec_in.transpose(0,1),
                h_enc.transpose(0,1)
            ).transpose(0,1)

        # MAE decoding (simply predict only masked positions; for brevity, not fully detailed)
        pred_mask = dec_out[:, mask_idx, :]
        target_mask = e[:, mask_idx, :]
        print("Predicted mask shape:", pred_mask.shape, "Target mask shape:", target_mask.shape)
        loss_mae = F.mse_loss(pred_mask, target_mask)

        return loss_mae

###################################################################################
# =================================================================================
# Example instantiation code (to copy/paste into your own training script):
#
# patchify_args = {
#     "in_channels": 1,
#     "hidden_dims": [128, 256, 512, 512],
#     "kernel_sizes": [4, 4, 4, 4],
#     "strides": [2, 2, 2, 2],
#     "lstm_hidden": 512,
#     "output_dim": 256,
# }
# encoder_args = {"embed_dim": 256, "n_heads": 8, "n_layers": 6}
# decoder_args = {"embed_dim": 512, "n_heads": 8, "n_layers": 6}
# mae_decoder_args = {"n_layers": 4}
# vq_args = {
#     "num_layers": 3,
#     "codebook_size": 2048,
#     "embed_dim": 256,
#     "semantic_prior": None,  # load your precomputed k-means centroids here
# }
# ar_args = {"embed_dim": 256, "n_heads": 8, "n_layers": 4}
# unpatchify_args = {
#     "output_channels": 1,
#     "hidden_dims": [512, 512, 256, 128],
#     "kernel_sizes": [4, 4, 4, 4],
#     "strides": [2, 2, 2, 2],
#     "lstm_hidden": 512,
#     "input_dim": 256,
# }
#
# model = ALMTokenizer(
#     patchify_args=patchify_args,
#     encoder_args=encoder_args,
#     decoder_args=decoder_args,
#     mae_decoder_args=mae_decoder_args,
#     vq_args=vq_args,
#     ar_args=ar_args,
#     unpatchify_args=unpatchify_args,
#     window_size=6,
# )
# print(model)
#
# =================================================================================
###################################################################################
