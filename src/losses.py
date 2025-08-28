import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio

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


def MultiScaleSpectrogramLoss(x_hat, x, scales=range(5, 12), alpha_per_scale=None, power=1.0, eps=1e-8):
    scales = list(scales)
    alpha = {i: 1.0 for i in scales}
    if alpha_per_scale is not None:
        alpha.update(alpha_per_scale)

    # Build one Spectrogram transform per scale
    losses = []
    for i in scales:
        n = 2 ** i
        S = torchaudio.transforms.MelSpectrogram(
            n_mels=64,
            sample_rate=24000,
            n_fft=n,
            window_fn=torch.hann_window,
            win_length=n,
            hop_length=n // 4,
            power=power,     # 1.0 -> magnitude, 2.0 -> power
            normalized=True,
            center=True,
            pad_mode="reflect",
            wkwargs={"device": x.device}
        ).to(x.device)

        X = S(x)                      # (B, C, F, T')
        Xh = S(x_hat)
        
        l1 = F.l1_loss(Xh, X)

        # logX = torch.log(X.abs() + eps)
        # logXh = torch.log(Xh.abs() + eps)
        l2 = torch.sqrt(((X - Xh)**2).mean(dim=-2)).mean()
        
        losses.append(l1 + alpha[i] * l2)

    return torch.stack(losses).mean()


class SubBandMultiScaleSpectrogramLoss(nn.Module):
    """
    Multiscale frequency-domain loss using *mel-scale band-split* subband spectrograms.

    ℓ_f(x, x̂) = (1/|S|) * Σ_{i∈S} ( 1/|B_i| * Σ_{b∈B_i} [ ||X_i^b(x̂) - X_i^b(x)||_1
                                                         + α_i * sqrt( MSE(·) + eps ) ] )
    where:
      - S = {5..11}, n_fft = 2**i, hop = n_fft//4, win = n_fft
      - B_i are subbands defined by mel-scale *boundaries* (no mel averaging!)
    """
    def __init__(self, scales=range(5, 12), alpha_per_scale=None, power=1.0, eps=1e-8,
                 sample_rate: int = 24000, n_mels: int = 64, mel_scale: str = "htk"):
        super().__init__()
        self.scales = list(scales)
        self.alpha = {i: 1.0 for i in self.scales}
        if alpha_per_scale is not None:
            self.alpha.update(alpha_per_scale)
        self.power = power          # 1.0 -> magnitude, 2.0 -> power
        self.register_buffer("eps_buf", torch.tensor(eps))
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel_scale = mel_scale

        # One Spectrogram (+ window) per scale
        self.specs = nn.ModuleDict()
        self.register_buffer("dummy", torch.empty(0))  # for device/dtype reference
        self.windows = nn.ParameterDict()  # store as buffers via Parameter with requires_grad=False
        for i in self.scales:
            n = 2 ** i
            # window per-scale (kept as parameter for device moves; no grad)
            w = torch.hann_window(n)
            self.windows[str(i)] = nn.Parameter(w, requires_grad=False)
            self.specs[str(i)] = torchaudio.transforms.Spectrogram(
                n_fft=n,
                win_length=n,
                hop_length=n // 4,
                power=self.power,      # magnitude or power on linear STFT
                normalized=True,
                center=True,
                pad_mode="reflect"
            )

        # Precomputed mel-based band edges (bin start/end) per scale
        # (pure indexing; stays on CPU)
        self.band_slices = {}  # key: str(i) -> list[(f0, f1)]
        with torch.no_grad():
            for i in self.scales:
                n = 2 ** i
                n_freqs = n // 2 + 1
                fb = torchaudio.functional.melscale_fbanks(
                    n_freqs=n_freqs,
                    f_min=0.0,
                    f_max=float(self.sample_rate) / 2.0,
                    n_mels=self.n_mels,
                    sample_rate=self.sample_rate,
                    norm=None,
                    mel_scale=self.mel_scale,
                )  # (n_freqs, n_mels)

                # For each mel filter, get span of STFT bins with nonzero weight
                bands = []
                fb_T = fb.T  # (n_mels, n_freqs)
                for m in range(self.n_mels):
                    nz = torch.nonzero(fb_T[m] > 0, as_tuple=False).squeeze(-1)
                    if nz.numel() == 0:
                        continue
                    f0 = int(nz.min().item())
                    f1 = int(nz.max().item()) + 1  # exclusive
                    if f1 > f0:
                        bands.append((f0, f1))
                # Merge duplicates (adjacent mels can map to same [f0,f1))
                merged = []
                for a0, a1 in bands:
                    if not merged or (a0, a1) != merged[-1]:
                        merged.append((a0, a1))
                self.band_slices[str(i)] = merged

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        x_hat, x: (B, 1, N)
        Returns a scalar loss averaged over scales and subbands.
        """
        assert x.shape == x_hat.shape and x.dim() == 3 and x.size(1) == 1, "Expect (B,1,N)"
        total_per_scale = []
        eps = self.eps_buf.to(x.dtype).to(x.device)

        for i in self.scales:
            S = self.specs[str(i)]
            w = self.windows[str(i)].to(device=x.device, dtype=x.dtype)
            bands = self.band_slices[str(i)]

            # Linear spectrograms: (B, 1, F, T)
            X  = S(x, window=w)
            Xh = S(x_hat, window=w)

            band_losses = []
            for (f0, f1) in bands:
                X_b  = X[:, :, f0:f1, :]
                Xh_b = Xh[:, :, f0:f1, :]

                l1 = F.l1_loss(Xh_b, X_b)
                mse = F.mse_loss(Xh_b, X_b)
                l2 = torch.sqrt(mse + eps)
                band_losses.append(l1 + self.alpha[i] * l2)

            if len(band_losses) == 0:
                continue  # extremely unlikely, but be safe for tiny FFTs

            scale_loss = torch.stack(band_losses).mean()
            total_per_scale.append(scale_loss)

        if len(total_per_scale) == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        return torch.stack(total_per_scale).mean()


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

def compute_adv_loss(logits_f: torch.Tensor) -> torch.Tensor:
    """
    Compute adversarial hinge loss for generator.
    Args:
        discriminators: List of Discriminator instances
        x_hat:          Generated waveform (B, 1, N)
        x:              Real waveform      (B, 1, N)
    Returns:
        Tensor scalar of adversarial loss (sum over discriminators).
    """
    
    K = len(logits_f)   # number of sub-discriminators

    # --- Adversarial (hinge-G using logit maps) ---
    # L_adv = (1/K) * sum_k mean( relu(1 - D_k(x_hat)) )
    L_adv = sum(F.relu(1.0 - fl).mean() for fl in logits_f) / K

    return L_adv


def compute_feat_loss(fmap_r: torch.Tensor, fmap_f: torch.Tensor) -> torch.Tensor:
    """
    Compute feature matching loss between real and fake for each discriminator.
    Args:
        discriminators: List of Discriminator instances
        x_hat:          Generated waveform (B, 1, N)
        x:              Real waveform      (B, 1, N)
    Returns:
        Tensor scalar of feature matching loss.
    """
    # --- relative feature matching (paper Eq.) ---
    L_feat = 0
    for dr, df in zip(fmap_r, fmap_f):  # per discriminator
        for rl, fl in zip(dr, df):
            L_feat += torch.mean(torch.abs(rl - fl) /
                         (torch.mean(torch.abs(rl))))
    L_feat = L_feat / (len(fmap_r) * len(fmap_r[0]))

    return L_feat


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
    losses['L_freq'] = lambdas["L_freq"] * MultiScaleSpectrogramLoss(x_hat, x)


    # Forward
    with torch.no_grad():
        logits_r, fmap_r = discriminators(x)       # lists of length K
    logits_f, fmap_f = discriminators(x_hat)
    
    # Adversarial
    losses['L_adv'] = lambdas["L_adv"] * compute_adv_loss(logits_f)
    losses['L_feat'] = lambdas["L_feat"] * compute_feat_loss(fmap_r, fmap_f)

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

def compute_discriminator_loss(discriminators: nn.ModuleList, x_fake, x_real):

        loss = 0.0

        try:
            K = len(discriminators)
        except TypeError:
            K = discriminators.num_discriminators
        
        # Logits shape: [B, C, T, F]
        real_logits, _ = discriminators(x_real)
        fake_logits, _ = discriminators(x_fake.detach())
        
        for rl, fl in zip(real_logits, fake_logits):

            # Hinge loss
            loss = loss + (F.relu(1.0 - rl).mean()
                + F.relu(1.0 + fl).mean())
        
        return loss / K