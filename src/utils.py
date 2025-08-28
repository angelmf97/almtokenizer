import torch
import torch.nn as nn
import torch.nn.functional as F

def mask_frames(frames, mask_rate=0.3, mask_token=None):
    """
    Masks a fraction of the frames in the input tensor.
    
    Args:
        frames (torch.Tensor): Input tensor of shape [B, T, D].
        mask_rate (float): Fraction of frames to mask.
        mask_token (torch.Tensor): Token to use for masking.
        
    Returns:
        torch.Tensor: Tensor with masked frames.
    """
    B, T, D = frames.shape
    num_mask = int(T * mask_rate)
    num_mask = max(num_mask, 1) if mask_rate > 0 else 0

    # Randomly choose mask positions   
    perm = torch.randperm(T, device=frames.device)
    mask_idx = perm[:num_mask].sort()[0]

    # Write mask token to the selected positions
    if mask_token is None:
        mask_token = nn.Parameter(torch.zeros(1, D, device=frames.device))
    
    masked_frames = frames.clone()
    masked_frames[:, mask_idx, :] = mask_token
    
    return masked_frames, mask_idx

def interleave_cls_tokens(frames, cls_token: torch.Tensor, window_size=2):

    B, T, D = frames.size()
    w = window_size
    
    # Number of CLS tokens = T // w
    rem = T % w

    if rem:
        frames = F.pad(frames, (0, 0, 0, w - rem, 0, 0), value=0)  # Pad with zeros to make T divisible by w
        T = frames.size(1) 

    n_cls = T // w

    # 1) Expand CLS tokens:
    cls_tokens = cls_token.repeat(B, n_cls, 1)  # (B, n_cls, embed_dim)

    # 2) Interleave: split frames into (B, n_cls, w, D)
    frames_ = frames.view(B, n_cls, w, D)  # (B, n_cls, w, D)
    cls_tokens_ = cls_tokens.unsqueeze(2)  # (B, n_cls, 1, D)

    interleaved = torch.cat([frames_, cls_tokens_], dim=2)  # (B, n_cls, w+1, D)
    interleaved = interleaved.view(B, -1, D)  # (B, T + n_cls, D)

    cls_positions = torch.tensor([(i + 1) * (w + 1) - 1 for i in range(n_cls)], device=frames.device).long()

    return interleaved, cls_positions

def interleave_mask_tokens(
    cls_tokens: torch.Tensor,
    window_size: int,
    mask_token: torch.Tensor
) -> torch.Tensor:
    """
    Interleave `window_size` copies of `mask_token` before each CLS token.

    Args:
        cls_tokens: Tensor of shape (B, n_cls, D)
        window_size: number of mask tokens to insert before each CLS
        mask_token: Tensor of shape (D,) or (1, D) or (B, 1, D)

    Returns:
        Tensor of shape (B, (window_size + 1) * n_cls, D)
    """
    B, n_cls, D = cls_tokens.shape

    # Prepare mask block of shape (B, n_cls, window_size, D)
    # Handle mask_token shapes
    mt = mask_token
    if mt.dim() == 1:                        # (D,)
        mt = mt.view(1, 1, D)
    if mt.shape == (1, D):                  # (1, D)
        mt = mt.unsqueeze(0)                # → (1, 1, D)
    if mt.shape == (1, 1, D):
        mt = mt.expand(B, n_cls, D)         # → (B, n_cls, D)
    if mt.shape == (B, 1, D):
        mt = mt.expand(B, n_cls, D)         # → (B, n_cls, D)
    else:
        # if it's already (B, n_cls, D), ok
        assert mt.shape == (B, n_cls, D), "mask_token must broadcast to (B, n_cls, D)"

    # Now expand to window_size
    mask_block = mt.unsqueeze(2).expand(B, n_cls, window_size, D)
    cls_block  = cls_tokens.unsqueeze(2)           # (B, n_cls, 1, D)

    # Concatenate [ masks..., CLS ] for each of the n_cls groups
    interleaved = torch.cat([mask_block, cls_block], dim=2)  # (B, n_cls, window_size+1, D)

    # Flatten the groups into one sequence
    B, n_cls, grp, D = interleaved.shape  # grp == window_size+1
    return interleaved.reshape(B, n_cls * grp, D)



def retrieve_cls_tokens(x, cls_positions):
    """
    Retrieves the CLS tokens from the interleaved tensor.
    
    Args:
        x (torch.Tensor): Interleaved tensor of shape [B, T + n_cls, D].
        cls_positions (list): List of positions of CLS tokens.
        
    Returns:
        torch.Tensor: Tensor containing only the CLS tokens.
    """
    B, T, D = x.size()

    cls_tokens = x[:, cls_positions, :]  # [B, n_cls, D]
    
    all_idx = torch.arange(T, device=x.device)
    frames_idx = all_idx[torch.isin(all_idx, cls_positions, invert=True)]
    frames = x[:, frames_idx, :]  # [B, T_frames, D]

    return cls_tokens, frames


import torchaudio.transforms as T

class MelLogMel(nn.Module):
    """
    Compute mel- and log-mel- spectrograms and stack them as two channels.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 128,
        win_length: int = 1024,
        n_mels: int = 128,
        top_db: float = 80.0,
    ):
        super().__init__()
        # Mel spectrogram: power spectrogram -> mel bins
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,                # power spectrogram
        )
        # Convert power spectrogram to decibels
        self.to_db = T.AmplitudeToDB(stype="power", top_db=top_db)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, N) or (B, N) waveform in [-1, 1]
        returns: (B, 2, n_mels, T) where channel 0 = mel, 1 = log-mel
        """
        # collapse channel if present
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)  # -> (B, N)
        # compute mel
        mel = self.mel_spec(x)       # (B, n_mels, T)
        log_mel = self.to_db(mel)    # (B, n_mels, T)
        # stack as two-channel feature
        return torch.stack([mel, log_mel], dim=1)
    

from .model import ALMTokenizer
from encodec.msstftd import MultiScaleSTFTDiscriminator

def load_model_from_config(cfg):

    device = torch.device(cfg["device"])

    cfg_model = cfg["model"]
    encoder_args      = cfg_model["encoder_args"]
    decoder_args      = cfg_model["decoder_args"]
    mae_decoder_args  = cfg_model["mae_args"]
    patchify_args     = cfg_model["patchify_args"]
    unpatchify_args   = cfg_model["unpatchify_args"]

    model = ALMTokenizer(
        device           = device,
        from_raw_audio   = True,
        encoder_args     = encoder_args,
        decoder_args     = decoder_args,
        mae_decoder_args = mae_decoder_args,
        patchify_args    = patchify_args,
        unpatchify_args  = unpatchify_args,
        window_size      = cfg["model"]["window_size"],
    ).to(device)

    return model

def load_discriminators_from_config(cfg):

    device = torch.device(cfg["device"])

    cfg_disc = cfg["discriminator"]
    hop_lengths = cfg_disc["hop_lengths"]
    n_fft = cfg_disc["n_fft"]
    win_lengths = cfg_disc["win_lengths"]
    filters = cfg_disc["filters"]
    filters_scale = cfg_disc["filters_scale"]
    dilations = cfg_disc["dilations"]
    max_filters = cfg_disc["max_filters"]

    discriminators = MultiScaleSTFTDiscriminator(
        filters=filters,
        filters_scale=filters_scale,
        max_filters=max_filters,
        n_ffts=n_fft,
        hop_lengths=hop_lengths,
        win_lengths=win_lengths,
        dilations=dilations,
    ).to(device)

    return discriminators
