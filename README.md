# almtokenizer


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
