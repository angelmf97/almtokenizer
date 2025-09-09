# ALMTokenizer - Custom Implementation

⚠️ This is an unofficial implementation of ALMTokenizer. It is not affiliated with the original authors.

This repository aims to reproduce the results of the paper ["ALMTokenizer: A Low-bitrate and Semantic-rich Audio Codec Tokenizer for Audio Language Modeling"](https://arxiv.org/abs/2504.10344). This work introduces a new way of compressing audio into discrete tokens that are both low-bitrate and semantically rich, making them more suitable for audio language modeling tasks such as text-to-speech, audio captioning, or music generation.

### Audio Examples

We carried out a series of experiments to compare EnCodec and ALMTokenizer across different tasks. First, we evaluated their ability to reconstruct audio signals, where EnCodec excels in fidelity while ALMTokenizer focuses on producing embeddings with stronger semantic organization. We then explored sound space traversals, where latent representations are gradually shifted between instruments to observe how timbre evolves. Finally, we tested zero-shot timbre transfer, attempting to change the instrument identity of a sound while preserving its pitch and rhythm.

You can listen to all these examples [here](https://angelmf97.github.io/almtokenizer/).


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







