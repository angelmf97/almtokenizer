# almtokenizer

This repository aims to reproduce the results of the paper ["ALMTokenizer: A Low-bitrate and Semantic-rich Audio Codec Tokenizer for Audio Language Modeling"](https://arxiv.org/abs/2504.10344). This work introduces a new way of compressing audio into discrete tokens that are both low-bitrate and semantically rich, making them more suitable for audio language modeling tasks such as text-to-speech, audio captioning, or music generation.


### Quick setup

Create and activate the conda environment using the following command:

```
conda env create -f environment.yml
conda activate almtokenizer
```

This environment uses python 3.10, torch 2.6.0 and CUDA toolkit 11.8 (a different version of CUDA toolkit may be needed depending on the machine).

### Training

A config file is provided in `config.yaml` to facilitate defining and tracking the parameters of several configurations. Choose the desired parameters and run the `training.ipynb` notebook. The training process can then be monitored by running:

```
tensorboard --logdir runs/base_model    # Default directory, can be changed in the config
```

The "scalars" tab allows to monitor the training and evaluation losses, and the "audio" tab allows to listen to a few pairs of original vs. reconstructed audios.

**Important:** Please make sure you have a copy of the FSD50K dataset and the path to the root directory of the dataset is correctly specified in the config file under dataloader/dataset_path.

The window size (separation between CLS tokens) changes randomly for each batch, in a range between 2 and 10, so the model generalizes better for different values of window size.

Most important parameters to check in the config:

```
model:
    base_args:
        **n_heads:** number of attention heads of the transformer encoder and decoder. In the original implementation they use 64 (probably beyond our limitations).
        **n_layers:** number of transformer layers of the transformer encoder and decoder. In the original implementation they use 24 (probably beyond our limitations).

    mae_args:
        n_heads: similar to base_args, but for the MAE decoder (not that many heads are needed).
        n_layers: similar to base_args, but for the MAE decoder (not that many layers are needed).

discriminator:
    hop_lengths, n_fft, win_lengths: self-explanatory, must be defined as lists. There must be as many values in each list as discriminators in the discriminator ensemble. Set to use 6 discriminators by default.
    n_mels: number of mel filters to compute for the spectrograms.

dataloader:
    **dataset_path:** must contain a path to the root directory of the FSD50K dataset.
    batch_size
    train_subset_size: for experimentation, allows to choose only a subset of the whole training dataset.
    test_subset_size: similar, for the test dataset.
    nsecs: desired length of the audios (in seconds) for trimming and padding audio samples. They use 5-second clips in the original implementation. Since the transformers implement a causal mask, that should be enough to generalize beyond 5-second audios.

training:
    num_epochs
    discriminator_train_freq: determines the interval (bumber of epochs) at which the discriminator is trained.
    checkpoint_freq: determines the interval (number of epochs) at which the checkpoints are saved.
    eval_freq: determines the interval (number of epochs) at which the losses for the evaluation set are computed.
    start_checkpoint: allows to start the training from a previous checkpoint. The weights will be loaded from this epoch from the checkpoint_dir (see below)
    writer_dir: defines where the tensorboard logs are stored.
    checkpoint_dir: defines where the generator and discriminator checkpoints are saved.
    lr_generator: learning rate of the generator.
    lr_discriminator: learning rate of the discriminator.
    lambdas: factors multiplying each term of the compound loss function of the generator.

```

### Sound Reconstruction Examples
<table>
  <tr>
    <td>    
        <p>Original:</p>
        <audio controls src="audio/speech-female.wav"></audio>
    </td>
    <td>
        <p>Reconstructed (w=3):</p>
        <audio controls src="audio/speech-female.wav"></audio>
    </td>
    <td>
        <p>Reconstructed (w=6):</p>
        <audio controls src="audio/speech-female.wav"></audio>
    </td>  
    <td>
        <p>Reconstructed (w=10):</p>
        <audio controls src="audio/speech-female.wav"></audio>
    </td>  
  </tr>
</table>


### Sound Space Traversals

<table>
    <tr>
        <p>From bass A3 to clarinet A6:</p>
        <td>    
            <p>EnCodec:</p>
            <audio controls src="EnCodec_trajectory_0.wav"></audio>
        </td>
        <td>
            <p>ALMTokenizer:</p>
            <audio controls src="ALMTokenizer_trajectory_0.wav"></audio>
        </td>  
    </tr>
    <tr>
        <p>From bass A3 to clarinet A6:</p>
        <td>    
            <p>EnCodec:</p>
            <audio controls src="EnCodec_trajectory_1.wav"></audio>
        </td>
        <td>
            <p>ALMTokenizer:</p>
            <audio controls src="ALMTokenizer_trajectory_1.wav"></audio>
        </td>  
    </tr>
    <tr>
        <p>From bass A3 to clarinet A6:</p>
        <td>    
            <p>EnCodec:</p>
            <audio controls src="EnCodec_trajectory_2.wav"></audio>
        </td>
        <td>
            <p>ALMTokenizer:</p>
            <audio controls src="ALMTokenizer_trajectory_2.wav"></audio>
        </td>  
    </tr>
</table>


### Zero-shot Timbre Transfer

In the following examples, we test whether we can change the timbre of a sound (its instrument-like quality) while keeping the pitch and rhythm intact.

The idea is simple:

1. We encode an input sound into latent representations using both EnCodec and ALMTokenizer.

2. For each instrument in the Good-sounds dataset, we compute an “average point” in latent space (a centroid).

3. To transform a sound, we take its latent representation and shift it toward the centroid of a target instrument.

4. Finally, we decode the shifted representation back into audio.

These examples show that while reconstruction quality is still limited, ALMTokenizer's latent space captures semantic structure more clearly. This makes the timbre transfer feel more intentional than with EnCodec, even if the results are far from perfect.

<table>
    <tr>
        <p>Female speech to cello:</p>
        <td>    
            <p>EnCodec before transfer:</p>
            <audio controls src="before_encodec_1.wav"></audio>
        </td>
        <td>
            <p>Encodec after transfer:</p>
            <audio controls src="after_encodec_1.wav"></audio>
        </td>
    </tr>
    <tr>
        <td>    
            <p>ALMTokenizer before transfer:</p>
            <audio controls src="before_encodec_1.wav"></audio>
        </td>
        <td>
            <p>ALMTokenizer after transfer:</p>
            <audio controls src="after_encodec_1.wav"></audio>
        </td>  
    </tr>
</table>
<table>
    <tr>
        <p>Male speech to cello:</p>
        <td>    
            <p>EnCodec before transfer:</p>
            <audio controls src="before_encodec_.wav"></audio>
        </td>
        <td>
            <p>Encodec after transfer:</p>
            <audio controls src="after_encodec_.wav"></audio>
        </td>
    </tr>
    <tr>
        <td>    
            <p>ALMTokenizer before transfer:</p>
            <audio controls src="before_encodec_.wav"></audio>
        </td>
        <td>
            <p>ALMTokenizer after transfer:</p>
            <audio controls src="after_encodec_.wav"></audio>
        </td>  
    </tr>
</table>



