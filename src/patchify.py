import torch
from encodec import EncodecModel

class Patchify:
    def __init__(self, device="cuda", model_name="encodec_24khz"):
        model = EncodecModel.encodec_model_24khz()
        self.encoder = model.encoder
        del model
        self.encoder.eval()
        self.device = device
        self.encoder.to(device)

    @torch.no_grad()
    def encode(self, wavs):
        wavs.to(self.device)
        x = self.encoder(wavs.to(self.device))
        return x
