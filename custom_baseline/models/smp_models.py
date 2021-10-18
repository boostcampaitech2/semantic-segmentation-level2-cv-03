import torch.nn as nn

import segmentation_models_pytorch as smp

class UnetPlusPlus(nn.Module):
    def __init__(self, num_classes=11):
        super(UnetPlusPlus, self).__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            )
    def forward(self, x):
        return self.model(x)