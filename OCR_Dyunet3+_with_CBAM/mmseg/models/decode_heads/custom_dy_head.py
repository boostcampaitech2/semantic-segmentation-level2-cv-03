from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .custom_dy.DyHead import DyHead
from .custom_dy.concat_fpn_output import concat_feature_maps
import torch.nn.functional as F
import numpy as np
import torch

@HEADS.register_module()
class CustomDyHead(BaseDecodeHead):
    def __init__(self,num_blocks,level=4,scale=48,**kwargs):
        super(CustomDyHead,self).__init__(**kwargs)
        self.L = level
        self.S = scale
        if isinstance(self.in_channels,int):
            self.C = self.in_channels
        else:
            self.C = self.in_channels[0]

        self.dy_head = DyHead(num_blocks,self.L,self.S**2,self.C)

    def forward(self,inputs):
        # B x channels x H x W
        x = self._transform_inputs(inputs) # transform style : multi select여야 한다.
        if not isinstance(x,list):
            x = [x] # to list
        assert len(x)==self.L

        levels=[]
        # heights = [ level.shape[2] for level in x]
        # median_height = int(np.median(heights))

        # assert median_height == self.S
        
        for i in range(len(x)):
            level = x[i]
            if level.shape[2] > self.S:
                level = F.interpolate(input=level, size=(self.S, self.S),mode='nearest')
            # If level height is less than median, then upsample
            else:
                level = F.interpolate(input=level, size=(self.S, self.S), mode='nearest')

            # # If level height is greater than median, then downsample with interpolate
            # if level.shape[2] > median_height:
            #     level = F.interpolate(input=level, size=(median_height, median_height),mode='nearest')
            # # If level height is less than median, then upsample
            # else:
            #     level = F.interpolate(input=level, size=(median_height, median_height), mode='nearest')
            levels.append(level)
        
        concat_levels = torch.stack(levels,dim=1) # B x level x C x H x W
        concat_levels = concat_levels.flatten(start_dim=3).transpose(dim0=2, dim1=3) # B x level x (H*W) x C

        output = self.dy_head(concat_levels) # B x level x (H*W) x C
        output = output.transpose(dim0=2,dim1=3).view(output.shape[0],output.shape[1],self.C,self.S,self.S) # B x level x C x H x W

        concat_output = [output[:,i,:,:,:] for i in range(output.shape[1])]
        concat_output = torch.cat(concat_output,dim=1) # B x C x H x W
        concat_output = self.cls_seg(concat_output) # B x 11 x H x W
        return concat_output
