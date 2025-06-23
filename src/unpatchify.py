import torch
from encodec import EncodecModel

class Unpatchify:
    def __init__(self, device="cuda", model_name="encodec_24khz"):
        model = EncodecModel.encodec_model_24khz()
        self.decoder = model.decoder
        self.decoder.eval()
        self.device = device
        self.decoder.to(device)

    @torch.no_grad()
    def decode(self, codes):
        codes = codes.to(self.device)
        # codes puede ser cuantizado o continuo según la instancia
        return self.decoder(codes)

