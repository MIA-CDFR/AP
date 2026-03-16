from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self,input_dim,n_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim,n_classes)

    def forward(self,x):
        return self.fc(x)
