import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ImageExpert(nn.Module):
    def __init__(self, num_classes=3):
        super(ImageExpert, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class TabularExpert(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(TabularExpert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)  