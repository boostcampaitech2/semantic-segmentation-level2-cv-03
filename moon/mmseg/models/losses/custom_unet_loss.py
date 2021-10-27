import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight
from .unet_loss.msssimLoss import MSSSIM
from .unet_loss.iouLoss import IOU
from .cross_entropy_loss import cross_entropy


@LOSSES.register_module()
class CustomUnetLoss(nn.Module):
    def __init__(self,
                class_weight=None,
                loss_weight=1.0,
                loss_name='loss_mssim_ce'):
        super(CustomUnetLoss,self).__init__()
        self._loss_name=loss_name
        self.loss_weight =loss_weight
        self.class_weight = get_class_weight(class_weight)

        self.mssim = MSSSIM()
        #self.iou = IOU()
        self.ce = cross_entropy
        
    def forward(self,cls_score,label,weight=None,**kwargs):
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        loss_cls = self.loss_weight * (self.mssim(cls_score,label) 
                #+ self.iou(cls_score,label)
                + self.ce(cls_score,label,weight,class_weight=class_weight,**kwargs))
        
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


