import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


class ContrastiveCNN(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        cnn_channels: int = 256,
        projection_dim: int = 128,
        temperature: float = 0.07,
        device: str = "cuda"
    ):
        """
        Red CNN que procesa secuencias de embeddings latentes (Z) de EnCodec
        con longitud variable, y aprende embeddings globales mediante pérdida InfoNCE.

        Parámetros:
        -----------
        latent_dim : int
            Dimensión de cada vector latente (por defecto 128D para EnCodec).
        cnn_channels : int
            Número de filtros en la capa Conv1D inicial (por defecto 256).
        projection_dim : int
            Dimensionalidad del espacio final donde se aplica InfoNCE (por defecto 128).
        temperature : float
            Parámetro de escala (tau) para la pérdida InfoNCE (por defecto 0.07).
        device : str
            Dispositivo donde se hospedan tensores ("cpu" o "cuda").
        """
        super(ContrastiveCNN, self).__init__()
        self.device = device
        self.temperature = temperature

        # 1. Bloque CNN 1D para extraer features temporales de latentes
        #    Entrada: (batch, latent_dim, T_max)
        #    Salida:  (batch, cnn_channels, T_max)  (preserva longitud T_max vía padding)
        self.conv1 = nn.Conv1d(
            in_channels=latent_dim, out_channels=cnn_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2. Segundo bloque Conv1D para aumentar campo receptivo y reducir resolución temporal
        #    Salida: (batch, cnn_channels, T_max//2)
        self.conv2 = nn.Conv1d(
            in_channels=cnn_channels, out_channels=cnn_channels,
            kernel_size=3, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm1d(cnn_channels)

        # 3. “Projection head”: reduce feature maps a un embedding global fijo
        #    Primero un AdaptiveAvgPool1d para obtener (batch, cnn_channels, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        #    Luego, MLP: cnn_channels → projection_dim (embedding final para InfoNCE)
        self.projection = nn.Sequential(
            nn.Linear(cnn_channels, cnn_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(cnn_channels // 2, projection_dim)
        )
        self.to(device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Paso hacia adelante (inference) de la CNN. Dado un lote de secuencias padded
        (batch, latent_dim, T_max) y su máscara (batch, T_max),
        retorna los embeddings de cada secuencia de tamaño [batch, projection_dim].

        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de latentes con forma (batch_size, latent_dim, T_max),
            donde T_max es la longitud máxima (con padding en secuencias cortas).
        mask : torch.Tensor, opcional
            Tensor booleano con forma (batch_size, T_max), donde mask[i, t]==1 indica
            frames válidos para la i-ésima secuencia; mask[i, t]==0 son posiciones de padding.
            Si es None, se asume que todas las posiciones son válidas.

        Retorna:
        --------
        torch.Tensor
            Embeddings normalizados L2 de forma (batch_size, projection_dim).
        """
        # x: (B, latent_dim, T_max)
        B, _, T_max = x.shape

        # 1. Primer bloque CNN + BN + ReLU
        h = self.conv1(x)            # → (B, cnn_channels, T_max)
        h = self.bn1(h)              # → (B, cnn_channels, T_max)
        h = self.relu(h)             # → (B, cnn_channels, T_max)

        # 2. Segundo bloque CNN + BN + ReLU (reduce temporal a T_max//2)
        h = self.conv2(h)            # → (B, cnn_channels, T_max//2)
        h = self.bn2(h)              # → (B, cnn_channels, T_max//2)
        h = self.relu(h)             # → (B, cnn_channels, T_max//2)

        # 3. Enmascarar posiciones de padding si se proporcionó una máscara
        if mask is not None:
            # Reducir la máscara a la mitad (stride=2), señalando frames válidos tras conv2
            mask = self._downsample_mask(mask)  # → (B, T_max//2)
            # Expandir máscara para coinceder con los canales
            mask = mask.unsqueeze(1).expand(-1, h.size(1), -1)  # → (B, cnn_channels, T_max//2)
            h = h.masked_fill(mask == 0, 0.0)                    # Ceros en posiciones de padding

        # 4. Global average pooling sobre dimensión temporal (celdas de frames válidos)
        #    Para evitar que los frames enmascarados (padding) contribuyan, usamos masked pool
        h = self._masked_global_pool(h, mask)  # → (B, cnn_channels)

        # 5. Proyección a espacio final
        z = self.projection(h)            # → (B, projection_dim)
        z = F.normalize(z, dim=1)         # L2-normalización para InfoNCE :contentReference[oaicite:3]{index=3}

        return z  # (B, projection_dim)

    def _downsample_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Reduce la máscara temporal por factor 2 (stride=2) para coincidir con conv2.
        Dado mask de (B, T_max), produce (B, T_max//2 + epsilon) booleano.
        Simplemente, para cada par de posiciones, si alguno es válido, la posición sigue siendo válida.
        """
        # Suponemos T_max par; si no, truncar el último frame
        B, T = mask.shape
        T2 = T // 2
        mask = mask[:, :T2*2]  # Truncar si T_max es impar
        mask = mask.reshape(B, T2, 2)
        # Si al menos uno de los dos frames originales era válido, mantenemos válido
        return (mask.sum(dim=2) > 0).to(torch.bool)  # → (B, T2)

    def _masked_global_pool(
        self, feats: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Realiza un global average pool en feats (B, C, T) ignorando posiciones enmascaradas.
        Si mask es None, procede como AdaptiveAvgPool1d normal.
        """
        B, C, T = feats.shape
        if mask is None:
            # Simple AdaptiveAvgPool1d
            return feats.mean(dim=2)  # → (B, C)

        # Sumar sobre frames válidos y dividir por número de frames válidos
        # feats * mask (convertida a float) deja valores en ceros donde mask==0
        float_mask = mask.float()  # (B, C, T)
        summed = (feats * float_mask).sum(dim=2)                 # → (B, C)
        denom  = float_mask.sum(dim=2).clamp(min=1e-6)            # → (B, C), evitar división por 0
        return summed / denom                                    # → (B, C)

    def info_nce_loss(
        self, z: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula InfoNCE loss para un lote de embeddings z y sus clases labels.
        z: (B, projection_dim), labels: (B,) enteros.
        Implementación basada en la definición InfoNCE:
          L = - (1/B) * Σ_i log [ exp(sim(z_i, z_i+)/T) / Σ_j exp(sim(z_i, z_j)/T) ]
        donde z_i+ es embedding de otro ejemplo con la misma etiqueta que i (positive).
        En nuestra versión supervisada, consideramos todos los con misma etiqueta como positivos.
        """
        B, D = z.shape
        # 1. Matriz de similitud: (B, B)
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # cos(z_i, z_j)/tau porque z está normalizado :contentReference[oaicite:4]{index=4}

        # 2. Máscara para pos/neg:
        labels = labels.contiguous().view(-1, 1)              # → (B, 1)
        mask_pos = torch.eq(labels, labels.T).float()         # → (B, B), 1 si i,j de misma clase
        # No queremos que cada embedding sea positivo consigo mismo en denominador
        # Así que ponemos cero en diagonal
        mask_self = torch.eye(B, device=self.device)
        mask_pos = mask_pos - mask_self                       # ahora diag=0, off-diag=1 para mismas clases 

        # 3. Para cada i, numerador = Σ_{j pos} exp(sim_{i,j})
        exp_sim = torch.exp(sim_matrix)                       # → (B, B)
        numerator = (exp_sim * mask_pos).sum(dim=1)            # → (B,)

        # 4. Denominador = Σ_{j≠i} exp(sim_{i,j})
        #    restar exp(sim_{i,i}) para excluirlo (o usar máscara)
        exp_sim_no_self = exp_sim * (1 - mask_self)           # pone cero en diagonal
        denominator = exp_sim_no_self.sum(dim=1)               # → (B,)

        # 5. Evitar log(0) con epsilon
        eps = np.finfo(np.float32).eps  # valor mínimo positivo para evitar log(0)
        loss = -torch.log((numerator + eps) / (denominator + eps))  # → (B,)
        return loss.mean()  # escalar por batch size :contentReference[oaicite:6]{index=6}

    def train_step(
        self,
        batch: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Realiza un paso de entrenamiento en la CNN con el lote dado.
        Parám:
          batch      : (B, latent_dim, T_max) tensor de secuencias padded
          mask       : (B, T_max) tensor booleano indicando frames válidos
          labels     : (B,) tensor de clases enteras para cada secuencia
          optimizer  : optimizador que actualiza parámetros del modelo
        Devuelve:
          loss.item() flotante del InfoNCE para este lote.
        """
        self.train()
        optimizer.zero_grad()
        z = self.forward(batch, mask)              # → (B, projection_dim)
        loss = self.info_nce_loss(z, labels)       # scalar
        loss.backward()
        optimizer.step()
        return loss.item()
