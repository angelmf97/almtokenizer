import torch
from torch.utils.data import Dataset
import torchaudio
import glob
import mirdata
import numpy as np
import bisect
import os

class JamendoDataset(Dataset):
    def __init__(self, split='train', sample_rate=24000, transform=None):
        # Carga la definición del dataset
        self.dataset = mirdata.initialize("mtg_jamendo_autotagging_moodtheme")
        # Filtra IDs según el split (mirdata no trae splits predefinidos;
        # aquí podrías partir aleatoriamente o usar metadata para train/val/test)
        all_ids = self.dataset.track_ids
        # Ejemplo: tomar 80% train, 10% val, 10% test
        n = len(all_ids)
        cutoff1 = int(0.8 * n)
        cutoff2 = int(0.9 * n)
        if split == 'train':
            self.track_ids = all_ids[:cutoff1]
        elif split == 'val':
            self.track_ids = all_ids[cutoff1:cutoff2]
        else:
            self.track_ids = all_ids[cutoff2:]
        self.sample_rate = sample_rate
        self.transform = transform

        tag_set = set()
        for tid in self.track_ids:
            # track.tags es una cadena, p. ej. "rock" 
            tag_str = self.dataset.track(tid).tags
            tag_set.add(tag_str)
        # 4. Ordenamos para asegurar determinismo y creamos mapeo tag→índice
        self.all_tags = sorted(tag_set)  # lista como ["ambient", "jazz", "pop", "rock", ...]
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.all_tags)}
        # Número total de clases
        self.num_classes = len(self.all_tags)


    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        track = self.dataset.track(track_id)
        wav_np, sr = track.audio           # numpy array [channels, T] :contentReference[oaicite:11]{index=11}
        wav = torch.from_numpy(wav_np)            # Tensor [C, T]
        
        # Aseguramos un canal
        if wav.dim() > 1:
            print(wav.shape)
            wav = wav.mean(dim=0)  # Convertimos a mono si es estéreo
            print(wav.shape)  
        # Resampleamos si hace falta (usando torchaudio)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(
                wav.unsqueeze(0), sr, self.sample_rate
            ).squeeze(0)                         # Volvemos a [2, T_resampled]

        wav = wav[:1000]

        # Add channel dimension:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # If stereo, turn it into mono by averaging channels:
        if wav.size(0) == 2:
            wav = wav.mean(dim=0, keepdim=True)

        # Creamos máscara completa (no hay padding en audio original):
        T = wav.size(1)
        mask = torch.ones(T, dtype=torch.bool)

        # Label (por ejemplo, el mismo track_id convertido a entero)
        tag_str = track.tags          # p. ej. "rock" 
        label_idx = self.tag2idx[tag_str]  # obtiene un entero :contentReference[oaicite:7]{index=7}
        label = torch.tensor(label_idx, dtype=torch.long)


        return wav, mask, label

class FSDKaggle2018Dataset(Dataset):
    """
    PyTorch Dataset for the FSDKaggle2018 audio dataset.

    This dataset expects a directory structure like:
        root_dir/
            FSDKaggle2018.audio_train/
                *.wav
            FSDKaggle2018.audio_test/
                *.wav

    Args:
        root_dir (str): Path to the dataset directory containing the two subfolders.
        split (str): One of 'train', 'test', or 'all'. Controls which subset to load.
        sample_rate (int): Target sample rate. Audio will be resampled to this rate if different.
        transform (callable, optional): A function/transform that takes in a waveform Tensor
            and returns a transformed version.
    """

    def __init__(self, root_dir, split='train', sample_rate=24000, transform=None):
        assert split in ('train', 'test', 'all'), \
            f"split must be 'train', 'test', or 'all', got {split}"
        self.root_dir = root_dir
        self.split = split
        self.sample_rate = sample_rate
        self.transform = transform

        # Collect all .wav file paths according to split
        self.file_paths = []
        if split in ('train', 'all'):
            train_pattern = os.path.join(root_dir, 'FSDKaggle2018.audio_train', '*.wav')
            self.file_paths.extend(glob.glob(train_pattern))
        if split in ('test', 'all'):
            test_pattern = os.path.join(root_dir, 'FSDKaggle2018.audio_test', '*.wav')
            self.file_paths.extend(glob.glob(test_pattern))

        if not self.file_paths:
            raise RuntimeError(f"No audio files found for split='{split}' in {root_dir}")

    def __len__(self):
        """Return the total number of audio files in this split."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load the waveform at index `idx`, convert to mono, resample, and apply transform.

        Returns:
            waveform (Tensor): FloatTensor of shape [1, T] where T is number of samples.
            sr (int): The sample rate of `waveform` (== self.sample_rate).
            filepath (str): Full path to the original WAV file.
        """
        filepath = self.file_paths[idx]
        waveform, sr = torchaudio.load(filepath)  # waveform shape: [channels, time]
        # Convert to mono if needed
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )

        # Apply user transform (e.g. augmentations, feature extraction)
        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform, self.sample_rate, filepath

class CodesPtDataset(Dataset):
    def __init__(self, files):
        self.files = files
        # 1) Precomputar número de muestras por fichero
        lengths = []
        for f in files:
            audio, tensor = torch.load(f, map_location='cpu')
            lengths.append(tensor.size(0))          # número de muestras en este batch-file

        # 2) Array de longitudes acumuladas para el mapeo rápido
        self.cum_lengths = np.cumsum(lengths)      # e.g. [16, 32, 48, ...]
        self.total_samples = int(self.cum_lengths[-1])

    def __len__(self):
        # Devuelve el total real de muestras
        return self.total_samples
    
    def __getitem__(self, idx):
        # 3) Encontrar el fichero que contiene la muestra idx
        file_idx = bisect.bisect_right(self.cum_lengths, idx)
        # 4) Calcular el índice dentro del batch-file
        prev_cum = self.cum_lengths[file_idx - 1] if file_idx > 0 else 0
        intra_idx = idx - prev_cum

        # 5) Cargar solo el fichero que toca y extraer la muestra concreta
        audio, codes = torch.load(self.files[file_idx], map_location='cpu')
        return audio[intra_idx], codes[intra_idx]
    

def collate_fn_codes(batch):
    """
    Collate function to pad a batch of waveforms to the same length.

    Args:
        batch (List[Tuple[Tensor, int, str]]):
            A list where each item is a tuple (waveform, sample_rate, filepath).
            - waveform: Tensor of shape [channels, T_i]
            - sample_rate: int
            - filepath: str

    Returns:
        padded_waveforms (Tensor): FloatTensor of shape [batch_size, channels, max_T]
        lengths (LongTensor): Tensor of shape [batch_size] with original lengths T_i
        sample_rates (List[int]): List of sample rates
        filepaths (List[str]): List of file paths
    """
    # Unzip the batch
    waveforms, codes = zip(*batch)

    # Determine original lengths and max length
    lengths = torch.tensor([w.shape[1] for w in codes], dtype=torch.long)
    max_length = lengths.max().item()

    # Assume all waveforms have the same number of channels
    channels = codes[0].shape[0]

    # Prepare a tensor of zeros for padding
    batch_size = len(codes)
    padded_codes = torch.zeros(batch_size, channels, max_length)

    # Copy each waveform into the padded tensor
    for i, c in enumerate(codes):
        length = c.shape[1]
        padded_codes[i, :, :length] = c

    return padded_codes

def collate_fn_audio(batch):
    """
    Collate function to pad a batch of waveforms to the same length.

    Args:
        batch (List[Tuple[Tensor, int, str]]):
            A list where each item is a tuple (waveform, sample_rate, filepath).
            - waveform: Tensor of shape [channels, T_i]
            - sample_rate: int
            - filepath: str

    Returns:
        padded_waveforms (Tensor): FloatTensor of shape [batch_size, channels, max_T]
        lengths (LongTensor): Tensor of shape [batch_size] with original lengths T_i
        sample_rates (List[int]): List of sample rates
        filepaths (List[str]): List of file paths
    """
    # Unzip the batch
    waveforms, sr, filepath = zip(*batch)

    # Determine original lengths and max length
    lengths = torch.tensor([w.shape[1] for w in waveforms], dtype=torch.long)
    max_length = lengths.max().item()
    if max_length > 24000*5:
        max_length = 24000*5

    # Assume all waveforms have the same number of channels
    channels = waveforms[0].shape[0]

    # Prepare a tensor of zeros for padding
    batch_size = len(waveforms)
    padded_waveforms = torch.zeros(batch_size, channels, max_length)

    # Copy each waveform into the padded tensor
    for i, c in enumerate(waveforms):
        length = min(c.shape[1], 24000*1)
        padded_waveforms[i, :, :length] = c[:, :length]

    return padded_waveforms
