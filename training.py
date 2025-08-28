
if __name__ == "__main__":
    # ### Load the Config File

    import yaml
    import shutil
    import os

    # config_path = "checkpoints/smallmodelmediumdisc/config_medium.yaml"
    config_path = "config_medium.yaml"


    # Load the YAML
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

        os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
        shutil.copy(config_path, cfg["training"]["checkpoint_dir"])

    # %% [markdown]
    # ### Define the DataLoader

    # %%
    import torch
    from torch.utils.data import DataLoader
    from src.datasets import FSD50K, collate_fn_audio
    from torch.utils.data import Subset

    dl_cfg = cfg["dataloader"]
    batch_size = dl_cfg["batch_size"]
    num_workers = dl_cfg["num_workers"]
    nsecs = dl_cfg["nsecs"]
    shuffle = dl_cfg["shuffle"]
    train_subset_size = dl_cfg["train_subset_size"]
    test_subset_size = dl_cfg["test_subset_size"]
    dataset_path = dl_cfg["dataset_path"]

    train_dataset = FSD50K(dataset_path, split="train")
    test_dataset = FSD50K(dataset_path, split="test")

    if train_subset_size is None:
        train_subset_size = int(len(train_dataset))

    if test_subset_size is None:
        test_subset_size = int(len(test_dataset))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(123)

    train_dl = DataLoader(Subset(train_dataset, range(train_subset_size)), 
                    batch_size=batch_size, 
                    shuffle=shuffle,
                    num_workers=num_workers,
                    collate_fn=lambda x: collate_fn_audio(x, nsecs=nsecs))

    test_dl = DataLoader(Subset(test_dataset, range(test_subset_size)), 
                    batch_size=batch_size, 
                    shuffle=shuffle,
                    num_workers=num_workers,
                    collate_fn=lambda x: collate_fn_audio(x, nsecs=nsecs))

    # %% [markdown]
    # ### Define the Model and Discriminators

    # %%
    import yaml
    from src.utils import load_model_from_config, load_discriminators_from_config
    from encodec.msstftd import MultiScaleSTFTDiscriminator
    import torchaudio

    from encodec.msstftd import DiscriminatorSTFT
    from encodec.modules.conv import NormConv2d
    from encodec.msstftd import get_2d_padding
    import typing as tp
    import torch.nn as nn

    def patched_init(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                    n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024, max_filters: int = 1024,
                    filters_scale: int = 1, kernel_size: tp.Tuple[int, int] = (3, 9), dilations: tp.List = [1, 2, 4],
                    stride: tp.Tuple[int, int] = (1, 2), normalized: bool = True, norm: str = 'weight_norm',
                    activation: str = 'LeakyReLU', activation_params: dict = {'negative_slope': 0.2}):
        torch.nn.Module.__init__(self)
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window,
            normalized=self.normalized, center=False, pad_mode=None, power=None)
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(spec_channels, self.filters, kernel_size=kernel_size, padding=get_2d_padding(kernel_size))
        )
        in_chs = self.filters
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride,
                                            dilation=(dilation, 1), padding=get_2d_padding(kernel_size, (dilation, 1)),
                                            norm=norm))
            in_chs = out_chs
        out_chs = min((filters_scale ** (len(dilations) + 1)) * self.filters, max_filters)
        self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=(kernel_size[0], kernel_size[0]),
                                        padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                        norm=norm))
        self.conv_post = NormConv2d(out_chs, self.out_channels,
                                    kernel_size=(kernel_size[0], kernel_size[0]),
                                    padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                    norm=norm)

    DiscriminatorSTFT.__init__ = patched_init


    model = load_model_from_config(cfg)
    discriminators = load_discriminators_from_config(cfg)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {model_params:,}")

    disc_params = sum(p.numel() for p in discriminators.parameters() if p.requires_grad)
    print(f"Discriminators parameters: {disc_params:,}")

    print(f"Total parameters: {model_params + disc_params:,}")

    # %% [markdown]
    # ### Train the Model

    # %%
    training_cfg = cfg["training"]
    num_epochs = training_cfg["num_epochs"]
    discriminator_train_freq = training_cfg["discriminator_train_freq"]
    d_train_prob = training_cfg["d_train_prob"]
    checkpoint_freq = training_cfg["checkpoint_freq"]
    eval_freq = training_cfg["eval_freq"]
    start_checkpoint = training_cfg["start_checkpoint"]
    writer_dir = training_cfg["writer_dir"]
    checkpoint_dir = training_cfg["checkpoint_dir"]
    checkpoint_dir = training_cfg["checkpoint_dir"]
    lr_generator = training_cfg["lr_generator"]
    lr_discriminator = training_cfg["lr_discriminator"]
    weight_decay = training_cfg["weight_decay"]
    betas = training_cfg["betas"]
    lambdas = training_cfg["lambdas"]

    model.train_model(train_dl=train_dl,
                        test_dl=test_dl,
                        discriminators=discriminators,
                        num_epochs=num_epochs,
                        discriminator_train_freq=discriminator_train_freq,
                        d_train_prob=d_train_prob,
                        checkpoint_freq=checkpoint_freq,
                        start_checkpoint=start_checkpoint,
                        eval_freq=eval_freq,
                        writer_dir=writer_dir,
                        checkpoint_dir=checkpoint_dir,
                        lr_g=lr_generator, 
                        weight_decay=weight_decay,
                        lr_d=lr_discriminator,
                        betas=betas,
                        lambdas=lambdas)


