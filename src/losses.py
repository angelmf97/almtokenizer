import torch
import torch.nn.functional as F
import torch.nn as nn

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
    # Normalize predictions and targets
    return F.mse_loss(mae_pred, mae_target)


def compute_adv_feat_losses(discriminators: nn.ModuleList, x_hat: torch.Tensor, x: torch.Tensor) -> tuple:
    """
    Compute adversarial hinge loss for generator.
    Args:
        discriminators: List of Discriminator instances
        x_hat:          Generated waveform (B, 1, N)
        x:              Real waveform      (B, 1, N)
    Returns:
        Tensor scalar of adversarial loss (sum over discriminators).
    """
    """
    Returns (L_adv, L_feat) for the generator.
    - Hinge-G (paper style): E[max(0, 1 - D(fake))]
    - Feature matching: mean L1 over layers per discriminator, then mean over K
    Assumes D(x) -> (logits_list, features_list) with one entry per sub-discriminator.
    """

    # Forward
    real_logits_list, real_fmaps_list = discriminators(x)       # lists of length K
    fake_logits_list, fake_fmaps_list = discriminators(x_hat)

    K = len(fake_logits_list)   # number of sub-discriminators

    # --- Adversarial (hinge-G using logit maps) ---
    # L_adv = (1/K) * sum_k mean( relu(1 - D_k(fake_map)) )
    L_adv = sum(F.relu(1.0 - fl).mean() for fl in fake_logits_list) / K

    # --- relative feature matching (paper Eq.) ---
    L_feat = 0.0
    eps = 1e-8
    for rf_k, ff_k in zip(real_fmaps_list, fake_fmaps_list):
        # rf_k / ff_k are lists of feature maps for sub-D k
        Lk = len(rf_k)
        perD = 0.0
        for r, f in zip(rf_k, ff_k):
            r_det = r.detach()
            num = (f - r_det).abs().mean()          # ||D_k^l(x̂) - D_k^l(x)||_1 mean over all dims
            den = r_det.abs().mean() + eps          # mean(||D_k^l(x)||_1)
            perD += num / den
        L_feat += perD / Lk
    L_feat /= K

    return L_adv, L_feat

def compute_adv_loss(discriminators: nn.ModuleList, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
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
    K = discriminators.num_discriminators
    for D in discriminators:

        fake_logits = D(x_hat)
        # hinge loss: E[max(0, 1 - D(x_hat))]
        loss += torch.mean(F.relu(1.0 - fake_logits))
    return loss / K


def compute_feat_loss(discriminators: nn.ModuleList, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
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
    K = discriminators.num_discriminators
    sample_feats_real = discriminators[0].get_features(x)
    L = len(sample_feats_real)

    for D in discriminators:
        feats_real = D.get_features(x)
        feats_fake = D.get_features(x_hat)
        for fr, ff in zip(feats_real, feats_fake):
            loss += F.l1_loss(ff, fr)
    
    normalized_loss = loss / (K * L)

    return normalized_loss


def compute_all_losses(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    discriminators: nn.ModuleList,
    lambdas: dict[str, float],
    mae_pred: torch.Tensor = None,
    mae_target: torch.Tensor = None,
    mask_idx: torch.LongTensor = None,
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
    losses['L_time'] = lambdas["L_time"] * compute_time_loss(x_hat, x)
    losses['L_freq'] = lambdas["L_freq"] * compute_freq_loss(x_hat, x)

    # Adversarial
    losses['L_adv'], losses['L_feat']  = compute_adv_feat_losses(discriminators, x_hat, x)
    losses['L_adv'] = lambdas["L_adv"] * losses['L_adv']
    losses['L_feat'] = lambdas["L_feat"] * losses['L_feat']

    # MAE
    if mae_pred is not None and mae_target is not None and mask_idx is not None:
        losses['L_mae'] = lambdas["L_mae"] * compute_mae_loss(mae_pred, mae_target, mask_idx)
    else:
        losses['L_mae'] = torch.tensor(0.0, device=x.device)
    # Total
    losses['L_total'] = (
        losses['L_time'] +
        losses['L_freq'] +
        losses['L_adv']  +
        losses['L_feat'] +
        losses['L_mae']
    )
    return losses

def compute_discriminator_loss(discriminators: nn.ModuleList, x_real, x_fake):

        loss = 0.0

        try:
            K = len(discriminators)
        except TypeError:
            K = discriminators.num_discriminators

        real_logits, _ = discriminators(x_real)
        fake_logits, _ = discriminators(x_fake.detach())
        
        for rl, fl in zip(real_logits, fake_logits):

            # Hinge loss
            loss = loss + (F.relu(1.0 - rl).mean()
                + F.relu(1.0 + fl).mean())
        
        return loss / K


def compute_discriminator_loss_fb(
    ms_stft_discriminator: torch.nn.Module,
    x_real: torch.Tensor,
    x_fake: torch.Tensor
) -> torch.Tensor:
    """
    Calcula la pérdida del discriminador usando hinge loss multiescala.

    Args:
        ms_stft_discriminator: discriminador MS-STFT.
        x_real: tensor de audio real, shape [B, 1, T].
        x_fake: tensor de audio generado, shape [B, 1, T].

    Returns:
        loss: escalar con la pérdida promedio sobre todos los discriminadores.
    """
    # Pasada por los discriminadores
    logits_real, _ = ms_stft_discriminator(x_real)
    logits_fake, _ = ms_stft_discriminator(x_fake.detach())

    # Acumulamos la pérdida
    loss_D = 0.0
    K = len(logits_real)
    for D_real_k, D_fake_k in zip(logits_real, logits_fake):
        # hinge loss: real --> max(0, 1 - D(x)), fake --> max(0, 1 + D(x̂))
        loss_real_k = F.relu(1.0 - D_real_k).mean()
        loss_fake_k = F.relu(1.0 + D_fake_k).mean()
        loss_D += (loss_real_k + loss_fake_k)

    return loss_D / K
