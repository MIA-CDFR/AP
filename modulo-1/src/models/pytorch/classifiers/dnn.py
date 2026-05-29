from torch import nn


class DNNClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.binary_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.family_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes - 1),
        )

    def forward(self, x):
        x = self.shared_backbone(x)
        x = self.layer2(x)
        binary_output = self.binary_head(x)
        family_output = self.family_head(x)
        return binary_output, family_output
