import torch
from encodec import EncodecModel

class Unpatchify:
    """
    A wrapper class for the Encodec decoder, providing an interface to decode audio representations.

    Attributes:
        decoder: The encoder part of the Encodec model.
        parameters: List of encoder parameters.
        device: The device (CPU or CUDA) to run the encoder on.
    """
    def __init__(self, device: str, model_name="encodec_24khz"):
        """
        Initializes the Unpatchify class by loading the Encodec decoder and moving it to the specified device.

        Args:
            device (str): The device to use ('cpu' or 'cuda').
            model_name (str): The name of the Encodec model to use (default: "encodec_24khz").
        """
        # Load the Encodec model (default: 24kHz)
        model = EncodecModel.encodec_model_24khz()

        # Extract the decoder from the model
        self.decoder = model.decoder
        self.decoder.train()
        self.parameters = list(self.decoder.parameters())
        self.device = device
        self.decoder.to(device)

    def decode(self, codes):
        """
        Decodes input waveforms using the Encodec decoder.

        Args:
            wavs (torch.Tensor): Input audio codes.

        Returns:
            torch.Tensor: Waveform reconstruction of the latent representation.
        """
        codes = codes.to(self.device)
        # codes puede ser cuantizado o continuo según la instancia
        return self.decoder(codes)

