import h5py
from encodec import EncodecModel
import torch
from tqdm import trange

def save_codes_h5(
    dataset,
    model,
    h5_path: str,
    device: str = 'cpu'
) -> None:
    """
    Encode each waveform with Encodec and save codes and metadata to an HDF5 file.

    Args:
        dataset (Dataset): GoodSoundsDataset instance.
        h5_path (str): Path to output .h5 file.
        device (str): torch device for model (e.g. 'cpu' or 'cuda').
    """
    # Load Encodec encoder
    encodec = EncodecModel.encodec_model_24khz().to(device)
    encoder = encodec.encoder

    with h5py.File(h5_path, 'w') as h5f:
        i = 0
        for idx in trange(len(dataset)):
            sample = dataset[-idx]
            wav = sample['waveform'].unsqueeze(0).to(device)
            if sample["klass"] == "good-sound":
                if sample["sustain"] is not None:
                    init = int(sample["sustain"]/2)
                elif sample["decay"] is not None:
                    init = int(sample["decay"]/2)
                elif sample["attack"] is not None:
                    init = int(sample["attack"]/2)
                else:
                    continue

                if sample["release"] is not None:
                    end = int(sample["release"]/2)
                
                elif sample["offset"] is not None:
                    end = int(sample["offset"]/2)
                else:
                    continue
                
                try:
                    wav = wav[:, :, init:end]
                except Exception as e:
                    print(f"Error processing sample {sample['sound_id']}: {e}")
                    print(init, end)
                    continue

            # Create a group for this sample
            grp_name = f"sound_{sample['sound_id']}_take_{sample['take_id']}"
            grp = h5f.create_group(grp_name)

            # Save metadata as attributes
            for key in ['sound_id', 'take_id', 'instrument', 'note', 'octave', 'dynamics', 'sustain', 'release', 'klass']:
                value = sample.get(key, None)
                # None -> empty string
                if value is None:
                    value = ''
                # Cast numeric types to int
                elif isinstance(value, torch.Tensor):
                    value = value.item()
                # Ensure all values are native Python types
                elif isinstance(value, (bytes, bytearray)):
                    value = value.decode('utf-8')
                else:
                    value = str(value)
                grp.attrs.create(key, value)



            with torch.no_grad():
                # Codes: list of [B, L] tensors for each codebook
                z_codes = encoder(wav)
                # Save each code tensor as a dataset
                for level, c in enumerate(z_codes):
                    # Remove batch dim
                    arr = c.squeeze(0).cpu().detach().numpy()
                    grp.create_dataset(f"z_level_{level}", data=arr, compression='gzip')

                h_codes = model.encode(wav)
                # Save each code tensor as a dataset
                for level, c in enumerate(h_codes):
                    # Remove batch dim
                    arr = c.squeeze(0).cpu().detach().numpy()
                    grp.create_dataset(f"h_level_{level}", data=arr, compression='gzip')

    print(f"Saved codes and metadata for {len(dataset)} samples to {h5_path}")

import os
import torchaudio
from tqdm import tqdm

def process_good_sounds_dataset(db, model, encodec=None, device="cpu", trim=False):
    if encodec is None:
        encodec = EncodecModel.encodec_model_24khz().to(device)
    model.to(device)
    encodec_latents = list()
    alm_latents = list()
    encodec_attributes = list()
    alm_attributes = list()
    db_root = "../good_sounds_dataset/"
    for idx in tqdm(db.index):
        start = 0
        end = -1

        if trim:

            #Prioritize sustain > decay > attack for the start
            if db.loc[idx, "sustain"]:
                start = int(db.loc[idx, "sustain"] // 2)
            elif db.loc[idx, "decay"]:
                start = int(db.loc[idx, "decay"] // 2)
            elif db.loc[idx, "attack"]:
                start = int(db.loc[idx, "attack"] // 2)

            if db.loc[idx, "release"]:
                end = int(db.loc[idx, "release"] // 2)
            elif db.loc[idx, "offset"]:
                end = int(db.loc[idx, "offset"] // 2)

        path = os.path.join(db_root, "good-sounds", db.loc[idx, "filename"])
        audio, sr = torchaudio.load(path)

        audio = audio[:, start:end]
        audio = audio.mean(dim=0, keepdim=True).unsqueeze(0).to(device)

        with torch.no_grad():
            # EnCodec
            encoded = encodec.encoder(audio)
            encoded = encoded.permute(0, 2, 1).squeeze(0)
            encodec_latents.append(encoded)
            attr_dict = db.loc[idx, ["instrument", "note", "octave", "klass"]].to_dict()
            encodec_attributes += [attr_dict] * encoded.shape[0]

            # ALM
            try:
                alm = model.encode(audio)
            except Exception as e:
                print(f"Error encoding audio {idx} with ALM: {e}")
                print(audio.shape)
                continue
            alm = alm.squeeze(0)
            alm_latents.append(alm)
            alm_attributes += [attr_dict] * alm.shape[0]
    return encodec_latents, encodec_attributes, alm_latents, alm_attributes

import pandas as pd

def aggregate_latents(encodec_latents, encodec_attributes, alm_latents, alm_attributes):
    z = torch.cat(encodec_latents, dim=0)
    h = torch.cat(alm_latents, dim=0)

    z_attr = pd.DataFrame(encodec_attributes)
    h_attr = pd.DataFrame(alm_attributes)

    assert z.shape[0] == z_attr.shape[0] 
    assert h.shape[0] == h_attr.shape[0]

    X = pd.concat([pd.DataFrame(z.cpu()), pd.DataFrame(h.cpu())], keys=["EnCodec", "ALMTokenizer"])
    df = pd.concat([z_attr, h_attr], keys=["EnCodec", "ALMTokenizer"])

    return X, df

import numpy as np

def create_vectors(trajectory, df, x):
    keys = list(trajectory.keys())
    Ls = [len(trajectory[k]) for k in keys]
    if len(set(Ls)) != 1:
        raise ValueError(f"All trajectory lists must be the same length; got lengths {dict(zip(keys, Ls))}")

    vectors = []
    for i in range(len(trajectory[keys[0]])):
        elem_masks = []
        for key in keys:
            elem_masks.append(df[key] == trajectory[key][i])
        reduced_mask = np.logical_and.reduce(elem_masks)
        samples = x[reduced_mask].to_numpy()
        centroid = samples.mean(axis=0)
        vectors.append(centroid)
    return vectors

def interpolate_latent(vectors, n):
    """
    Linearly interpolate n steps between vectors h1 and h2.

    Args:
        h1 (array-like, shape (D,)): Start latent vector.
        h2 (array-like, shape (D,)): End latent vector.
        n (int): Number of interpolation points (including h1 and h2).

    Returns:
        np.ndarray of shape (n, D): Interpolated vectors.
    """
    interpolated = []
    for i in range(len(vectors) - 1):
        h1 = np.asarray(vectors[i])
        h2 = np.asarray(vectors[i + 1])
        # 1. Create n weights from 0.0 to 1.0
        alphas = np.linspace(0.0, 1.0, num=int(n/len(vectors)))
        # 2. Compute each interpolation: (1-α)*h1 + α*h2
        interpolated.append(torch.tensor([(1.0 - a) * h1 + a * h2 for a in alphas]).squeeze(1))
    
    return torch.cat(interpolated)

def align_centroid(starts, dest):
    """
    Translate all starting vectors so their centroid equals dest.

    Args:
        starts (array-like, shape (M, D)): Starting vectors.
        dest   (array-like, shape (D,)):   Destination vector.

    Returns:
        np.ndarray, shape (M, D): Translated vectors whose centroid == dest.
    """

    delta = dest - starts.mean(axis=0)      # direction+amount to move the centroid
    return starts + delta  

def align_centroid_torch(starts, dest):
    X = starts.to(dtype=torch.float32)           # (M, D)
    d = dest.to(X.device, dtype=torch.float32)             # (D,)
    delta = d - X.mean(dim=(0, 1))                    # (D,)
    return X + delta                             # (M, D)

from IPython.display import Audio, display

def timbre_transfer(model, wav_path, move_to, X, df, device="cuda"):
    waveform, sr = torchaudio.load(wav_path, normalize=True)
    waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sr, new_freq=24000
                )

    wav_tensor = torch.tensor(waveform, dtype=torch.float32).mean(dim=0, keepdim=True)[None, :].to(device)

    with torch.no_grad():
        # EnCodec
        encodec = EncodecModel.encodec_model_24khz().eval().to(device)
        encoder = encodec.encoder.to(device)
        decoder = encodec.decoder.to(device)

        frames = encoder(wav_tensor.to(device)).permute(0, 2, 1)
        destination = create_vectors(move_to, df.loc["EnCodec"], X.loc["EnCodec"])
        destination = torch.tensor(destination[0])
        new_frames = align_centroid_torch(frames, destination)
        decoder_input = new_frames.permute(0, 2, 1)
        decoder_output = decoder(decoder_input)
        wav_encodec = decoder_output.squeeze(0)
        before_encodec = encodec(wav_tensor).squeeze(0)
        display(Audio(before_encodec.flatten().cpu().detach().numpy(), rate=24000)) 
        display(Audio(wav_encodec.flatten().cpu().detach().numpy(), rate=24000))

        # ALMTokenizer
        frames = model.encode(wav_tensor.to(model.device))
        destination = create_vectors(move_to, df.loc["ALMTokenizer"], X.loc["ALMTokenizer"])
        destination = torch.tensor(destination[0])
        new_frames = align_centroid_torch(frames, destination)
        decoder_input = new_frames
        decoder_output = model.decode(decoder_input)
        wav_alm = decoder_output.squeeze(0)
        before_alm = model(wav_tensor.to(model.device))["x_hat"].squeeze(0)
        display(Audio(before_alm.flatten().cpu().detach().numpy(), rate=24000)) 
        display(Audio(wav_alm.flatten().cpu().detach().numpy(), rate=24000))
    return before_encodec, wav_encodec, before_alm, wav_alm

from sklearn.model_selection import train_test_split
def stratified_sample(df, source_col="source", n=100, group_cols=("instrument", "note", "octave", "klass"), random_state=0):
    df = df.copy()
    # Create a combined label for stratification
    
    sample_indices = []
    for src, d in df.groupby(level=0):
        strata = d[list(group_cols)].astype(str).agg("_".join, axis=1)
        # If not enough rows, just take all
        n_target = min(n, len(d))
        if n_target == len(d):
            sample_indices += d
        else:
            # Stratified downsample
            s, _ = train_test_split(
                d, train_size=n_target, 
                stratify=strata, 
                random_state=random_state
            )
            sample_indices += s.index.tolist()

    return sample_indices

from sklearn.preprocessing import LabelEncoder

def generate_label_encoder(df, keys):
    encoders = dict()
    num_labels = df.copy()
    for key in keys:
        le = LabelEncoder()
        le.fit(df[key])
        encoders[key] = le
        num_labels[key] = le.transform(df[key])
    return encoders, num_labels

def get_good_bad(label: str) -> str:
    if not label or label.strip() == "":
        return "other"
    if "good" in label:
        return "good"
    if "bad" in label:
        return "bad"
    return "other"

from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
from matplotlib import cm

def projection(x, labels_df, columns, proj_fn=TSNE, y=None, legend=False, proj_fn_kwargs={}, transform_kwargs={}, plot_kwargs={}):

    proj = proj_fn(**proj_fn_kwargs)
    
    
    if y is None:
        coords = proj.fit_transform(x, **transform_kwargs)

    fig, axs = plt.subplots(1, len(columns), figsize=(20, 5))
    try:
        axs = axs.flatten()
    except AttributeError:
        axs = [axs]

    labels = labels_df
    for ax, key in zip(axs, columns):
        if y is not None:
            y_grp = y[key]
            coords = proj.fit_transform(x, y_grp, **transform_kwargs)
        colors = cm.get_cmap("tab10", len(labels_df[key].unique()))
        if len(labels_df[key].unique()) > 10:
            colors = cm.get_cmap("tab20", len(labels_df[key].unique()))
        for label, c in zip(labels_df[key].unique(), colors.colors.tolist()):
            # Get coordinates for the current label
            grp_coords = coords[labels[key] == label]
            ax.scatter(grp_coords[:, 0], grp_coords[:, 1], color=c, s=1.5, cmap="tab10", label=label)
        ax.set_title(key)
        xlabel = plot_kwargs.get("xlabel", "Axis 1")
        ax.set_xlabel(xlabel)

        ylabel = plot_kwargs.get("ylabel", "Axis 2")
        ax.set_ylabel(ylabel)

        if legend:
            ax.legend(loc='upper right', fontsize='small', markerscale=2, ncol=2)
        ax.set_aspect("equal")

    if "suptitles" in plot_kwargs:
        suptitle = plot_kwargs["suptitles"]
        fig.suptitle(suptitle, fontsize=16, y=1.1)
    return fig
