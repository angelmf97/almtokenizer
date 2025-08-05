import torch
from encodec import EncodecModel

class Patchify:
    """
    A wrapper class for the Encodec encoder, providing an interface to encode audio waveforms.

    Attributes:
        encoder: The encoder part of the Encodec model.
        parameters: List of encoder parameters.
        device: The device (CPU or CUDA) to run the encoder on.
    """
    def __init__(self, device: str, requires_grad: bool = False, model_name="encodec_24khz"):
        """
        Initializes the Patchify class by loading the Encodec encoder and moving it to the specified device.

        Args:
            device (str): The device to use ('cpu' or 'cuda').
            model_name (str): The name of the Encodec model to use (default: "encodec_24khz").
        """
        # Load the Encodec model (default: 24kHz)
        model = EncodecModel.encodec_model_24khz()
        
        # Extract the encoder from the model
        self.encoder = model.encoder
        
        del model
        self.encoder.train()
        self.parameters = list(self.encoder.parameters())
        self.device = device
        self.encoder.to(device)
        # Set requires_grad for the parameters
        for param in self.parameters:
            param.requires_grad = requires_grad

    def encode(self, wavs):
        """
        Encodes input waveforms using the Encodec encoder.

        Args:
            wavs (torch.Tensor): Input audio waveforms.

        Returns:
            torch.Tensor: Encoded representation of the input waveforms.
        """
        wavs.to(self.device)
        x = self.encoder(wavs)
        return x
