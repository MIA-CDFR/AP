
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self,input_dim,embed_dim=128,hidden_dim=128,n_classes=6):
        super().__init__()
        self.embedding = nn.Linear(input_dim,embed_dim)
        self.lstm = nn.LSTM(embed_dim,hidden_dim,batch_first=True)
        self.fc = nn.Linear(hidden_dim,n_classes)

    def forward(self,x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        output,(h,c) = self.lstm(x)
        return self.fc(h[-1])
