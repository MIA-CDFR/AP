import torch
import os


class PytorchUtils:
    @property
    def prefers_mps(self):
        return os.getenv("PYTORCH_ENABLE_MPS", "0") == "1"

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")

        if self.prefers_mps and torch.backends.mps.is_available():
            try:
                if torch.backends.mps.is_built():
                    return torch.device("mps")
            except AttributeError:
                return torch.device("mps")

        return torch.device("cpu")

torch_utils = PytorchUtils()
