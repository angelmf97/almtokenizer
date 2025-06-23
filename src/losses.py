import torch
import torch.nn.functional as F

from .discriminator import Discriminator


def compute_time_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 loss in the time domain between reconstructed and original waveforms.
    Args:
        x_hat: Reconstructed waveform, shape (B, 1, N)
        x:     Original waveform, shape (B, 1, N)
    Returns:
        Tensor scalar of time-domain L1 loss.
    """
    return F.l1_loss(x_hat, x)


def default_stft(x: torch.Tensor,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = None,
                 window: torch.Tensor = torch.hamming_window(1024)) -> torch.Tensor:
    """
    Compute magnitude spectrogram using PyTorch's STFT.
    Args:
        x:          Waveform tensor, shape (B, 1, N)
        n_fft:      FFT size
        hop_length: Hop length
        win_length: Window length (defaults to n_fft)
    Returns:
        Complex STFT tensor of shape (B, freq_bins, time_frames)
    """
    # Remove channel dim
    x = x.squeeze(1)  # (B, N)
    if win_length is None:
        win_length = n_fft
    # Compute STFT returning complex tensor
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window.to(x.device),
        center=True,
        return_complex=True
    )  # (B, freq_bins, time_frames)
    return spec


def compute_freq_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 loss in frequency domain between reconstructed and original waveforms.
    Uses default_stft internally.
    Args:
        x_hat: Reconstructed waveform, shape (B, 1, N)
        x:     Original waveform, shape (B, 1, N)
    Returns:
        Tensor scalar of frequency-domain L1 loss.
    """
    X_hat = default_stft(x_hat)
    X     = default_stft(x)
    # Magnitude comparison
    return F.l1_loss(torch.abs(X_hat), torch.abs(X))


def compute_mae_loss(mae_pred: torch.Tensor, mae_target: torch.Tensor, mask_idx: torch.LongTensor) -> torch.Tensor:
    """
    Compute MAE MSE loss on masked frame embeddings.
    Args:
        mae_pred:   Predicted embeddings, shape (B, T, D)
        mae_target: Original embeddings, shape (B, T, D)
        mask_idx:   1D indices of masked positions
    Returns:
        Tensor scalar of MSE loss over masked positions.
    """
    return F.mse_loss(mae_pred, mae_target)


def compute_adv_loss(discriminators: list[Discriminator], x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute adversarial hinge loss for generator.
    Args:
        discriminators: List of Discriminator instances
        x_hat:          Generated waveform (B, 1, N)
        x:              Real waveform      (B, 1, N)
    Returns:
        Tensor scalar of adversarial loss (sum over discriminators).
    """
    loss = 0.0
    for D in discriminators:

        fake_logits = D(x_hat.detach())
        # hinge loss: E[max(0, 1 - D(x_hat))]
        loss += torch.mean(F.relu(1.0 - fake_logits))
    return loss


def compute_feat_loss(discriminators: list[Discriminator], x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute feature matching loss between real and fake for each discriminator.
    Args:
        discriminators: List of Discriminator instances
        x_hat:          Generated waveform (B, 1, N)
        x:              Real waveform      (B, 1, N)
    Returns:
        Tensor scalar of feature matching loss.
    """
    loss = 0.0
    K = len(discriminators)
    sample_feats_real = discriminators[0].get_features(x)
    L = len(sample_feats_real)

    for D in discriminators:
        feats_real = D.get_features(x)
        feats_fake = D.get_features(x_hat)
        for fr, ff in zip(feats_real, feats_fake):
            loss += F.l1_loss(ff, fr)
    
    normalized_loss = loss / (K * L)

    return normalized_loss


def compute_generator_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    discriminators: list[Discriminator],
    mae_pred: torch.Tensor = None,
    mae_target: torch.Tensor = None,
    mask_idx: torch.LongTensor = None,
    lambda_mae: float = .5
) -> dict[str, torch.Tensor]:
    """
    Compute all generator losses and return dictionary.
    Args:
        x_hat: reconstructed waveform
        x:     original waveform
        discriminators: list of discriminators for adversarial loss
        mae_pred, mae_target, mask_idx: outputs for MAE loss (optional)
        lambda_feat: weight for feature matching loss
        lambda_mae:  weight for MAE loss
    Returns:
        dict containing 'L_time', 'L_freq', 'L_adv', 'L_feat', 'L_mae', 'L_total'
    """
    losses = {}
    # Reconstruction
    losses['L_time'] = compute_time_loss(x_hat, x)
    losses['L_freq'] = compute_freq_loss(x_hat, x)

    # Adversarial
    losses['L_adv']  = compute_adv_loss(discriminators, x_hat, x)
    losses['L_feat'] = compute_feat_loss(discriminators, x_hat, x)

    # MAE
    if mae_pred is not None and mae_target is not None and mask_idx is not None:
        losses['L_mae'] = compute_mae_loss(mae_pred, mae_target, mask_idx)
    else:
        losses['L_mae'] = torch.tensor(0.0, device=x.device)
    # Total
    losses['L_total'] = (
        losses['L_time'] +
        losses['L_freq'] +
        losses['L_adv']  +
        losses['L_feat'] +
        lambda_mae * losses['L_mae']
    )
    return losses

def compute_discriminator_loss(discriminators: list[Discriminator], x_real, x_fake):

        loss = 0.0
        feat_loss = 0.0
        K    = len(discriminators)
        for D in discriminators:

            # Hinge loss
            loss = loss + (F.relu(1.0 - D(x_real)).mean()
                + F.relu(1.0 + D(x_fake)).mean())
        
        return loss / K
