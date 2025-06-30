import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .patchify import Patchify
from .unpatchify import Unpatchify
from .transformers import QueryEncoder, QueryDecoder, PositionalEncoding
from .utils import interleave_cls_tokens, retrieve_cls_tokens, interleave_mask_tokens, mask_frames

from typing import Iterable
import torch.optim as optim
from src.losses import compute_generator_loss, compute_discriminator_loss
from itertools import chain
from typing import Optional
import os

from tqdm import trange, tqdm
from torch.utils.tensorboard.writer import SummaryWriter


class ALMTokenizer(nn.Module):
    """
    Combines:
      • Patchify (Encodec‐style),
      • QueryEncoder,
      • UnPatchify (Encodec‐style),
      • MAEDecoder (QueryDecoder with MAE loss).
    Example usage (sketch):
        model = ALMTokenizer(patchify_args, encoder_args, decoder_args,
                              mae_args, unpatchify_args, window_size=6)
        x_hat = model(x)
        compute losses (reconstruction, adversarial, MAE, etc.)
    """

    def __init__(
        self,
        encoder_args: dict,
        decoder_args: dict,
        mae_decoder_args: dict,
        patchify_args: dict,
        unpatchify_args: dict,
        from_raw_audio: bool = False,
        window_size: int = 6,
        device="cuda"
    ):
        super().__init__()
        self.from_raw_audio = from_raw_audio
        self.window_size = window_size
        self.device = device
        
        self.patchify = Patchify(**patchify_args)
        self.unpatchify = Unpatchify(**unpatchify_args)
        
        self.query_encoder = QueryEncoder(**encoder_args)
        self.query_decoder = QueryEncoder(**decoder_args)

        self.mae_decoder = QueryDecoder(**mae_decoder_args)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_args["embed_dim"]), requires_grad=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_args["embed_dim"]), requires_grad=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the encoder part of the model.
        """
        if self.from_raw_audio:
            frames = self.patchify.encode(x)  # (B, T, D)

        else:
            frames = x

        frames = frames.permute(0, 2, 1)  # (B, D, T) for Transformer compatibility
        
        B, T, D = frames.shape

        # Interleave CLS tokens
        cls_frames, cls_positions = interleave_cls_tokens(frames, 
                                       cls_token=self.cls_token,
                                       window_size=self.window_size) 
        
        # Encoder
        enc_out = self.query_encoder(cls_frames)  # (B, T + n_cls, D)

        # Retrieve CLS tokens
        h, _ = retrieve_cls_tokens(enc_out, cls_positions=cls_positions)

        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Returns the decoder part of the model.
        """
    
        # Interleave mask tokens before each CLS token
        cls_and_mask = interleave_mask_tokens(
            h, 
            window_size=self.window_size, 
            mask_token=self.mask_token
        )   # (B, (window_size + 1) * n_cls, D)

        # Decode masked frames
        dec_out = self.query_decoder(cls_and_mask)  # (B, T + n_cls, D)
        
        # One CLS token per window, so we can retrieve CLS tokens
        cls_positions = torch.tensor([(i + 1) * (self.window_size + 1) - 1 for i in range(h.size(1) // (self.window_size + 1))], device=h.device).long()

        # Remove CLS tokens
        _, dec_out = retrieve_cls_tokens(dec_out, cls_positions=cls_positions)  # (B, T, D)

        # Unpatchify
        x_hat = self.unpatchify.decode(dec_out.permute(0, 2, 1)) # (B, 1, N_samples)

        return x_hat



    def forward(
        self, x: torch.Tensor, mask_rate: float = 0.3, w: int = None
    ) -> torch.Tensor:
        """
        x: (batch, 1, N_samples)
        mask_rate: for MAE masking in MAE training
        w: optional override for window_size
        returns: reconstructed waveform x_hat: (batch, 1, N_samples)
        """

        if self.from_raw_audio:
            frames = self.patchify.encode(x)  # (B, T, D)

        else:
            frames = x

        frames = frames.permute(0, 2, 1)  # (B, D, T) for Transformer compatibility
        
        B, T, D = frames.shape

        # Interleave CLS tokens
        cls_frames, cls_positions = interleave_cls_tokens(frames, 
                                       cls_token=self.cls_token,
                                       window_size=self.window_size) 
        
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
        dec_out = self.query_decoder(cls_and_mask)

        # Remove CLS tokens
        _, dec_out = retrieve_cls_tokens(dec_out, cls_positions=cls_positions)


        # Unpatchify
        x_hat = self.unpatchify.decode(dec_out.permute(0, 2, 1)) # (B, 1, N_samples)

        # Trim for wrong frame count caused by encodec
        if x_hat.size(2) > x.size(2):
            x_hat = x_hat[:, :, :x.size(2)]

        # MAE training
        if self.training:
            masked_frames, masked_idx = mask_frames(frames, mask_rate=mask_rate, mask_token=self.mask_token)

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
    
    def load_model(self, path):
        """
        Loads model weights from the specified file path.

        Args:
            path (str): Path to the checkpoint file.

        Returns:
            None
        """
        self.load_state_dict(torch.load(path, map_location=self.device))
        return None

    def train_model(
            self,
            discriminators,
            dl,
            lambdas: dict,
            num_epochs: int = 200,
            checkpoint_freq: int = 10,
            start_checkpoint: Optional[int] = 0,
            discriminator_train_freq: int = 30,
            writer_dir: str = "writer",
            checkpoint_dir: str = "checkpoints",
            lr_g: float = 1e-4,
            weight_decay: float = 1e-5,
            lr_d: float = 2e-5,
            betas: Iterable[float] = (0.5, 0.9),
            ):
        """
        Trains the ALMTokenizer model with reconstruction, adversarial and MAE losses.

        Args:
            discriminators: List of discriminator models.
            dl: DataLoader providing training batches.
            lambdas: Dictionary of loss weights.
            num_epochs: Number of training epochs.
            checkpoint_freq: Frequency (in epochs) to save checkpoints.
            start_checkpoint: Epoch to resume training from.
            discriminator_train_freq: Frequency (in epochs) to train discriminators.
            writer_dir: Directory for TensorBoard logs.
            checkpoint_dir: Directory to save checkpoints.
            lr_g: Learning rate for generator.
            weight_decay: Weight decay for generator optimizer.
            lr_d: Learning rate for discriminator.
            betas: Betas for Adam optimizer.
        """

        # Ensure output directories exist
        os.makedirs(writer_dir, exist_ok=False)
        os.makedirs(checkpoint_dir, exist_ok=False)

        # TensorBoard writer for logging
        writer = SummaryWriter(log_dir="runs/alm_tokenizer")

        # Optimizer for generator (the ALMTokenizer itself)
        optim_g = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr_g,
            weight_decay=weight_decay
        )

        # Optimizer for all discriminators
        optim_d = optim.Adam(
            params=chain(*[D.parameters() for D in discriminators]),
            lr=lr_d,
            betas=betas
        )

        # Optionally load checkpoint to resume training
        if start_checkpoint:
            self.load_model(os.path.join(checkpoint_dir, f"alm_tokenizer_epoch_{start_checkpoint}.pth"))

        self.train()

        # Training loop over epochs
        for epoch in trange(num_epochs, initial=start_checkpoint):
            
            # Initialize loss accumulators
            losses = {
                "L_time": 0.0,
                "L_freq": 0.0,
                "L_adv": 0.0,
                "L_feat": 0.0,
                "L_mae": 0.0,
                "L_total": 0.0
                }
            
            # Iterate over batches
            for wavs in dl:
                
                wavs = wavs.to(self.device)

                # Forward pass through model
                res = self(wavs)

                x_hat = res["x_hat"]
                x = res["orig_waveform"]
                mae_pred = res["mae_pred"]
                mae_target = res["mae_target"]
                mask_idx = res["mask_indices"]

                # Compute generator loss (reconstruction, adversarial, MAE, etc.)
                generator_loss = compute_generator_loss(
                    x_hat=x_hat,
                    x=x,
                    discriminators=discriminators,
                    mae_pred=mae_pred,
                    mae_target=mae_target,
                    lambdas = lambdas,
                    mask_idx=mask_idx
                )

                # Accumulate losses for logging
                for loss_type, loss_value in generator_loss.items():
                    losses[loss_type] = losses[loss_type] + loss_value.item()
                
                total_gen_loss = generator_loss["L_total"]

                # Backpropagation for generator
                optim_g.zero_grad()
                total_gen_loss.backward()
                optim_g.step()
            
                # Train discriminators at specified frequency
                if epoch % discriminator_train_freq == 0:
                    x_disc = x.detach()
                    x_hat_disc = x_hat.detach()
                    discriminator_loss = compute_discriminator_loss(discriminators, x_disc, x_hat_disc)

                    optim_d.zero_grad()
                    discriminator_loss.backward()
                    optim_d.step()

                    # Log discriminator loss
                    writer.add_scalar(f"losses/discriminators", discriminator_loss, epoch)
            
            # Save the model every checkpoint_freq epochs
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(self.state_dict(), os.path.join(checkpoint_dir, f"alm_tokenizer_epoch_{epoch + 1}.pth"))
                print(f"Model saved at epoch {epoch + 1}")    

            # Log average losses to TensorBoard
            for loss_type, loss_value in losses.items():
                losses[loss_type] /= len(dl)
                writer.add_scalar(f"losses/{loss_type}", losses[loss_type], epoch)

            # Free up GPU memory
            torch.cuda.empty_cache()


