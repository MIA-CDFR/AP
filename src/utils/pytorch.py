import torch


class PytorchUtils:
    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    

torch_utils = PytorchUtils()
