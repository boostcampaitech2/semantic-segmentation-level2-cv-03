import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..builder import NECKS
from .fpn import FPN
from .. import builder

@NECKS.register_module()
class CBFPN(FPN):
    '''
    FPN with weight sharing
    which support mutliple outputs from cbnet
    '''
    def forward(self, inputs):
        if not isinstance(inputs[0], (list, tuple)):
            inputs = [inputs]
        
        ## 수정된 코드
        ## CBNetV2의 outputs을 element-wise summation 진행
        if self.training:
            temp=[]
            for x in inputs:
                out = super().forward(x)
                temp.append(out)
            
            outs=[]
            for x in zip(*temp):
                outs.append(sum(x))
            return outs
        ###

        # if self.training:
        #     outs = []
        #     for x in inputs:
        #         out = super().forward(x)
        #         outs.append(out)
        #     return outs
        else:
            out = super().forward(inputs[-1])
            return out
