from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self,input_dim,n_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim,n_classes)

    def forward(self,x):
        return self.linear(x)
