import argparse
import os
import torch
from torch.utils.data import DataLoader
from datasets import JamendoDataset
from src.patchify import QuantizedEncoder, ContinuousEncoder
from src.unpatchify import QuantizedDecoder, ContinuousDecoder
from utils import save_audio, save_vectors

def process_batch(encoder, decoder, batch, names, out_dir, mode):
    vectors = encoder.encode(batch)
    save_vectors(vectors, names, out_dir, mode)
    audio_rec = decoder.decode(vectors)
    save_audio(audio_rec, names, out_dir, mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mode", choices=["quantized", "continuous"], required=True)
    args = parser.parse_args()

    ds = JamendoDataset(
        split=args.split,
        sample_rate=48000,
        transform=None
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    if args.mode == "quantized":
        encoder = QuantizedEncoder()
        decoder = QuantizedDecoder()
    else:
        encoder = ContinuousEncoder()
        decoder = ContinuousDecoder()

    os.makedirs(args.output_dir, exist_ok=True)

    for wavs, names in dl:
        process_batch(encoder, decoder, wavs, names, args.output_dir, args.mode)
