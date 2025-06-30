import torch
import torch.nn as nn

class Discriminator2(nn.Module):
    """
    Single‐scale discriminator (blocks of Conv1d + LeakyReLU).
    You would typically instantiate several with different hop lengths.
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_dims: list = [64, 128, 256, 512, 512, 512],
        kernel_sizes: list = [3, 3, 3, 3, 3, 3],
        strides: list = [1, 2, 2, 2, 2, 2],
        mel_transform: nn.Module = None
    ):
        super().__init__()
        self.mel_transform = mel_transform
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
        x = self.mel_transform(x)
        x = x.squeeze(1)  # remove channel dimension if present
        x = torch.log(torch.clamp(x, min=1e-5))
        return self.net(x)
    

    def get_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Return a list of feature‐maps from each Conv1d block, to be used
        in feature‐matching loss
        """
        feats = []
        x = self.mel_transform(x)
        x = x.squeeze(1)  # remove channel dimension if present
        x = torch.log(torch.clamp(x, min=1e-5))
        out = x
        for layer in self.net:
            out = layer(out)
            # collect activations right after each Conv1d
            if isinstance(layer, nn.Conv1d):
                feats.append(out)
        return feats


class ResBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, leaky_slope=0.2):
        super().__init__()
        pad = kernel_size // 2
        # main path
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=pad)
        self.lrelu = nn.LeakyReLU(leaky_slope)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, stride=stride, padding=pad)
        # skip connection to match shape
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.lrelu(out)
        out = self.conv2(out)
        return self.lrelu(out + identity)
    

class Discriminator(nn.Module):
    """
    Single‐scale 2D residual discriminator head.
    Stacks `n_blocks` of ResBlock2d, all with the same `hidden_dim`.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        n_blocks: int,
        block_cls: nn.Module = ResBlock2d,
        mel_transform: nn.Module = None
    ):
        super().__init__()
        assert mel_transform is not None, "Provide a MelLogMel instance"
        self.mel_transform = mel_transform

        layers = []
        ch = in_channels
        for _ in range(n_blocks):
            layers.append(block_cls(ch, hidden_dim, kernel_size=3, stride=2))
            ch = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, N) or (B, N) waveform
        returns: flattened logits/features
        """
        spec = self.mel_transform(x)       # → (B, 2, n_mels, T)
        out = self.net(spec)               # → (B, hidden_dim, H, W)
        return out.flatten(1)              # → (B, hidden_dim * H * W)

    def get_features(self, x: torch.Tensor):
        """
        Return intermediate feature-maps after each ResBlock2d.
        """
        feats = []
        spec = self.mel_transform(x)
        out = spec
        for layer in self.net:
            out = layer(out)
            feats.append(out)
        return feats