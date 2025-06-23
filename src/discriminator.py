import torch
import torch.nn as nn

class Discriminator(nn.Module):
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