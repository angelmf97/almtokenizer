import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .patchify import Patchify
from .unpatchify import Unpatchify
from .transformers import QueryEncoder, QueryDecoder, PositionalEncoding
from .utils import interleave_cls_tokens, retrieve_cls_tokens, interleave_mask_tokens, mask_frames


#
# ──────────────────────────────────────────────────────────────────────────
#    FULL ALMTokenizer ASSEMBLY
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
        mae_encoder_args: dict,
        mae_decoder_args: dict,
        patchify_args: dict,
        unpatchify_args: dict,
        from_raw_audio: bool = False,
        window_size: int = 8,
    ):
        super().__init__()
        self.from_raw_audio = from_raw_audio
        self.window_size = window_size
        
        self.patchify = Patchify(**patchify_args)
        self.unpatchify = Unpatchify(**unpatchify_args)
        
        self.query_encoder = QueryEncoder(**encoder_args)
        self.query_decoder = QueryDecoder(**decoder_args)

        self.mae_encoder = QueryEncoder(**mae_encoder_args)
        self.mae_decoder = QueryDecoder(**mae_decoder_args)

        self.pos_encoder = PositionalEncoding(encoder_args["embed_dim"])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_args["embed_dim"]))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_args["embed_dim"]))



    def forward(
        self, x: torch.Tensor, mask_rate: float = 0.25, w: int = None
    ) -> torch.Tensor:
        """
        x: (batch, 1, N_samples)
        mask_rate: for MAE masking in stage1
        w: optional override for window_size
        returns: reconstructed waveform x_hat: (batch, 1, N_samples)
        """

        if self.from_raw_audio:
            frames = self.patchify.encode(x.to(self.patchify.device))  # (B, T_frames, encoder_dim)
            frames = frames.to("cuda")  # Ensure x is on the same device as patchify
        else:
            frames = x

        frames = frames.permute(0, 2, 1)  # (B, D, T) for Transformer compatibility
        
        B, T, D = frames.shape

        # Interleave CLS tokens
        cls_frames, cls_positions = interleave_cls_tokens(frames, 
                                       window_size=self.window_size, 
                                       cls_token=self.query_encoder.cls_token)
        
        # Encoder
        enc_out = self.query_encoder(cls_frames)  # (B, T + n_cls, D)

        # Retrieve CLS tokens
        h, _ = retrieve_cls_tokens(enc_out, cls_positions=cls_positions)

        # Interleave mask tokens before each CLS token
        cls_and_mask = interleave_mask_tokens(
            h, 
            window_size=self.window_size, 
            mask_token=self.mask_token
        )

        # Decode masked frames
        dec_out = self.query_decoder(cls_and_mask, cls_frames)

        # Remove CLS tokens
        _, dec_out = retrieve_cls_tokens(dec_out, cls_positions=cls_positions)


        # Unpatchify
        x_hat = self.unpatchify.decode(dec_out.permute(0, 2, 1).to(self.unpatchify.device))  # (B, 1, N_samples)
        x_hat = x_hat.to(x.device)  # Ensure x_hat is on the same device as x

        # Trim for wrong frame count caused by encodec
        if x_hat.size(2) > x.size(2):
            x_hat = x_hat[:, :, :x.size(2)]

        # MAE
        if self.training:
            masked_frames, masked_idx = mask_frames(frames, mask_rate=0.3, mask_token=self.mask_token)

            # Encoder
            mae_enc_out = self.query_encoder(masked_frames)  # (B, T, D)
            
            # Decoder
            mae_dec_out = self.mae_decoder(mae_enc_out, frames)

            pred_frames = mae_dec_out[:, masked_idx, :]  # (B, num_masked, D)
            original_frames = frames[:, masked_idx, :]

            return {
                "x_hat": x_hat,
                "orig_waveform": x,
                "mask_indices": masked_idx,
                "mae_pred": pred_frames,
                "mae_target": original_frames
            }


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
        frames = self.patchify.encode(x)  # (B, T_frames, encoder_dim)
        frames = frames.permute(0, 2, 1)  # (B, D, T) for Transformer compatibility
        B, T, D = frames.shape
 
        # 2) Masking
        masked_frames, masked_idx = mask_frames(frames, mask_rate=0.3, mask_token=self.mask_token)

        # Encoder
        enc_out = self.query_encoder(masked_frames)  # (B, T, D)
        
        # Decoder
        dec_out = self.mae_decoder(enc_out, frames)

        pred_frames = dec_out[:, masked_idx, :]  # (B, num_masked, D)
        original_frames = frames[:, masked_idx, :]  # (B, num_masked, D)
        loss_mae = F.mse_loss(pred_frames, original_frames)
        return loss_mae

