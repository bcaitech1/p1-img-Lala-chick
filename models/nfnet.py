from torch import nn
import timm


class MaskNFNet(nn.Module):
    def __init__(self, model_arch, n_classes, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            model_arch, pretrained=pretrained, num_classes=n_classes
        )
        n_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(n_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x
