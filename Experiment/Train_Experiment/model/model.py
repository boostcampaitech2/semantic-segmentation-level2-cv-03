
# (c) ConvMixer https://github.com/tmp-iclr/convmixer/blob/main/convmixer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


def conv_block(in_ch, out_ch, k_size, stride, padding, dilation=1, relu=True):
    block = []
    block.append(nn.Conv2d(in_ch, out_ch, k_size, stride, padding, dilation, bias=False))
    block.append(nn.BatchNorm2d(out_ch))
    if relu:
        block.append(nn.ReLU())
    return nn.Sequential(*block)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, in_chans=3, num_classes=1000, activation=nn.GELU, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dim
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size),
            activation(),
            nn.BatchNorm2d(dim)
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        activation(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    activation(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
          
    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        #x = self.pooling(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        #x = self.head(x)

        return x

class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block1 = conv_block(in_ch, out_ch, 1, 1, padding=0)
        self.block2 = conv_block(in_ch, out_ch, 3, 1, padding=6, dilation=6)
        self.block3 = conv_block(in_ch, out_ch, 3, 1, padding=12, dilation=12)
        self.block4 = conv_block(in_ch, out_ch, 3, 1, padding=18, dilation=18)
        self.block5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        
    def forward(self, x):
        upsample_size = (x.shape[-1], x.shape[-2])
        
        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        out5 = self.block5(x)
        out5 = F.interpolate(
            out5, size=upsample_size, mode="bilinear", align_corners=False
        )
        
        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        return out

class DeepLabV3(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.backbone = ConvMixer(1536, 20, num_classes=n_classes)
        self.aspp = AtrousSpatialPyramidPooling(1536, 256) #default= (2048, 256)
        self.conv1 = conv_block(256*5, 256, 1, 1, 0)
        self.conv2 = nn.Conv2d(256, n_classes, kernel_size=1)
        
    def forward(self, x):
        upsample_size = (x.shape[-1], x.shape[-2])

        backbone_out = self.backbone(x)
        aspp_out = self.aspp(backbone_out)
        out = self.conv1(aspp_out)
        out = self.conv2(out)
        
        out = F.interpolate(
            out, size=upsample_size, mode="bilinear", align_corners=True
        )
        return out
