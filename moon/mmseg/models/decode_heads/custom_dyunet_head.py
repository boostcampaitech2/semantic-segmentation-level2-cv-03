import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from mmcv.cnn import MaxPool2d
from mmcv.cnn import Conv2d

from ..builder import HEADS
from .decode_head import BaseDecodeHead

from .custom_dy.DyHead import DyHead

@HEADS.register_module()
# for FPN output
class CustomDyUnetHead(BaseDecodeHead):
    def __init__(self,
                feature_scale=4,
                num_blocks=6,
                scale=128,
                is_deconv=True,
                is_batchnorm=True,
                **kwargs):
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        super(CustomDyUnetHead,self).__init__(**kwargs)
        
        assert len(self.in_channels)==4
        # resolution : [128x128, 64x64, 32x32, 16x16]
        filters = self.in_channels # [512,512,512,512]

        self.CatChannels = filters[0] # 512
        self.CatBlocks = len(filters) # 4
        self.UpChannels = self.CatChannels * self.CatBlocks # 512 * 4


        self.h1_PT_hd4 = MaxPool2d(8, 8, ceil_mode=True) # 128/8 x 128/8
        self.h2_PT_hd4 = MaxPool2d(4, 4, ceil_mode=True) # 64/4 x 64/4
        self.h3_PT_hd4 = MaxPool2d(2, 2, ceil_mode=True) # 32/2 x 32/2

        self.conv4d_1 = ConvModule(self.UpChannels, self.UpChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)# 16
        

        '''stage 3d'''
        self.h1_PT_hd3 = MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd3 = MaxPool2d(2, 2, ceil_mode=True)
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)

        self.conv3d_1 = ConvModule(self.UpChannels, self.UpChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)  # 16


        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = MaxPool2d(2, 2, ceil_mode=True)
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)

        self.conv2d_1 = ConvModule(self.UpChannels, self.UpChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)  # 16

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation


        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)

        self.conv1d_1 = ConvModule(self.UpChannels, self.UpChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)  # 16

        # -------------Bilinear Upsampling--------------
        self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # DeepSup
        self.outconv1 = Conv2d(self.UpChannels, self.num_classes, 3, padding=1)
        self.outconv2 = Conv2d(self.UpChannels, self.num_classes, 3, padding=1)
        self.outconv3 = Conv2d(self.UpChannels, self.num_classes, 3, padding=1)
        self.outconv4 = Conv2d(self.UpChannels, self.num_classes, 3, padding=1)

        self.cls = nn.Sequential(
                    nn.Dropout(p=0.5),
                    Conv2d(filters[3],self.num_classes, 1),
                    nn.AdaptiveMaxPool2d(1),
                    nn.Sigmoid())

        ## dyhead
        self.L_size = 4
        self.S_size = scale
        self.C_size = self.CatChannels
        self.dy_head = DyHead(num_blocks,L=self.L_size,S=self.S_size**2,C=self.C_size)
        self.dy_conv1 = Conv2d(self.C_size,self.num_classes,3,padding=1)
        self.dy_conv2 = Conv2d(self.C_size,self.num_classes,3,padding=1)
        self.dy_conv3 = Conv2d(self.C_size,self.num_classes,3,padding=1)
        self.dy_conv4 = Conv2d(self.C_size,self.num_classes,3,padding=1)

    
    def dotProduct(self,seg,cls):
        # cls : B x 11 
        # seg : B x 11 x 128 x 128 
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def forward(self, inputs):
        ## -------------Encoder-------------
        x = self._transform_inputs(inputs) # transform style : multi select여야 한다.
        h1,h2,h3,h4 = x
        
        # -------------Classification-------------
        ## custom
        cls_branch = self.cls(h4).squeeze(3).squeeze(2) # (B,11)
        cls_branch_max = (cls_branch>=0.5).type(torch.int)

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4(h1)
        h2_PT_hd4 = self.h2_PT_hd4(h2)
        h3_PT_hd4 = self.h3_PT_hd4(h3)
        h4_Cat_hd4 = h4
        hd4 = self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4), 1))# hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3(h1)
        h2_PT_hd3 = self.h2_PT_hd3(h2)
        h3_Cat_hd3 = h3
        hd4_UT_hd3 = self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))
        hd3 = self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3), 1))# hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2(h1)
        h2_Cat_hd2 = h2
        hd3_UT_hd2 = self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))
        hd4_UT_hd2 = self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))
        hd2 = self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2), 1)) # hd2->160*160*UpChannels

        h1_Cat_hd1 = h1
        hd2_UT_hd1 = self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))
        hd3_UT_hd1 = self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))
        hd4_UT_hd1 = self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))
        hd1 = self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1), 1))# hd1->320*320*UpChannels

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) # B x 11 x 128 x 128

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) # B x 11 x 128 x 128

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2) # B x 11 x 128 x 128

        d1 = self.outconv1(hd1) # B x 11 x 128 x 128

        # -------------Dyhead--------------
        assert len(x)==self.L_size

        levels=[]
        #assert 128 == self.S_size
        
        for i in range(len(x)):
            level = x[i]
            if level.shape[-1] != self.S_size:
                level = F.interpolate(input=level, size=(self.S_size, self.S_size), mode='nearest')
            levels.append(level)
        
        concat_levels = torch.stack(levels,dim=1) # B x level x C x H x W
        concat_levels = concat_levels.flatten(start_dim=3).transpose(dim0=2, dim1=3) # B x level x (H*W) x C

        concat_output = self.dy_head(concat_levels) # B x level x (H*W) x C
        concat_output = concat_output.transpose(dim0=2,dim1=3).view(concat_output.shape[0],self.L_size,self.C_size,self.S_size,self.S_size)# B x level x C x H x W

        dy1,dy2,dy3,dy4 = [concat_output[:,i,:,:,:] for i in range(concat_output.shape[1])]
        dy1 = self.dy_conv1(dy1) # B x 11 x H x W
        dy2 = self.dy_conv1(dy2)
        dy3 = self.dy_conv1(dy3)
        dy4 = self.dy_conv1(dy4)

        # for ms train
        dy1 = F.interpolate(input=dy1,size=d1.shape[2:],mode='nearest') # B x 11 x in_H x in_W
        dy2 = F.interpolate(input=dy2,size=d2.shape[2:],mode='nearest')
        dy3 = F.interpolate(input=dy3,size=d3.shape[2:],mode='nearest')
        dy4 = F.interpolate(input=dy4,size=d4.shape[2:],mode='nearest')
    
        # -------------- output ---------
        out1 = self.dotProduct(d1+dy1,cls_branch_max) # B x classes x H x W
        out2 = self.dotProduct(d2+dy2,cls_branch_max)
        out3 = self.dotProduct(d3+dy3,cls_branch_max)
        out4 = self.dotProduct(d4+dy4,cls_branch_max)

        output = torch.cat((out1,out2,out3,out4),1)
        output = self.cls_seg(output)

        return output






# @HEADS.register_module()
# class CustomDyUnetHead(BaseDecodeHead):
#     def __init__(self,
#                 feature_scale=4,
#                 is_deconv=True,
#                 is_batchnorm=True,
#                 **kwargs):
#         self.is_deconv = is_deconv
#         self.is_batchnorm = is_batchnorm
#         self.feature_scale = feature_scale
#         super(CustomDyUnetHead,self).__init__(**kwargs)
        
#         assert len(self.in_channels)==4
#         # resolution : [128x128, 64x64, 32x32, 16x16]
#         filters = self.in_channels # [128,256,512,1024]

#         self.CatChannels = filters[0] # 128
#         self.CatBlocks = len(filters) # 4
#         self.UpChannels = self.CatChannels * self.CatBlocks # 512


#         self.h1_PT_hd4 = MaxPool2d(8, 8, ceil_mode=True) # 128/8 x 128/8
#         self.h1_PT_hd4_conv = ConvModule(filters[0], 
#                             self.CatChannels, 
#                             3, 
#                             padding=1,
#                             conv_cfg=self.conv_cfg,
#                             norm_cfg=self.norm_cfg,
#                             act_cfg=self.act_cfg) # 128

#         # h2->160*160, hd4->40*40, Pooling 4 times
#         self.h2_PT_hd4 = MaxPool2d(4, 4, ceil_mode=True) # 64/4 x 64/4
#         self.h2_PT_hd4_conv = ConvModule(filters[1],
#                             self.CatChannels,
#                             3,
#                             padding=1,
#                             conv_cfg=self.conv_cfg,
#                             norm_cfg=self.norm_cfg,
#                             act_cfg=self.act_cfg) # 128

#         # h3->80*80, hd4->40*40, Pooling 2 times
#         self.h3_PT_hd4 = MaxPool2d(2, 2, ceil_mode=True) # 32/2 x 32/2
#         self.h3_PT_hd4_conv = ConvModule(filters[2],
#                                          self.CatChannels,
#                                           3, 
#                                           padding=1,
#                                           conv_cfg=self.conv_cfg,
#                                           norm_cfg=self.norm_cfg,
#                                           act_cfg=self.act_cfg) # 128

#         # h4->40*40, hd4->40*40, Concatenation
#         self.h4_Cat_hd4_conv = ConvModule(filters[3],
#                                             self.CatChannels, 
#                                             3,
#                                             padding=1,
#                                             conv_cfg=self.conv_cfg,
#                                             norm_cfg=self.norm_cfg,
#                                             act_cfg=self.act_cfg)

#         self.conv4d_1 = ConvModule(self.UpChannels, self.UpChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)# 16
        

#         '''stage 3d'''
#         # h1->320*320, hd3->80*80, Pooling 4 times
#         self.h1_PT_hd3 = MaxPool2d(4, 4, ceil_mode=True)
#         self.h1_PT_hd3_conv = ConvModule(filters[0], self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)
        

#         # h2->160*160, hd3->80*80, Pooling 2 times
#         self.h2_PT_hd3 = MaxPool2d(2, 2, ceil_mode=True)
#         self.h2_PT_hd3_conv = ConvModule(filters[1], self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)
        

#         # h3->80*80, hd3->80*80, Concatenation
#         self.h3_Cat_hd3_conv = ConvModule(filters[2], self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)
        

#         # hd4->40*40, hd4->80*80, Upsample 2 times
#         self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
#         self.hd4_UT_hd3_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)
        
#         self.conv3d_1 = ConvModule(self.UpChannels, self.UpChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)  # 16

#         '''stage 2d '''
#         # h1->320*320, hd2->160*160, Pooling 2 times
#         self.h1_PT_hd2 = MaxPool2d(2, 2, ceil_mode=True)
#         self.h1_PT_hd2_conv = ConvModule(filters[0], self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)

#         # h2->160*160, hd2->160*160, Concatenation
#         self.h2_Cat_hd2_conv = ConvModule(filters[1], self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)

#         # hd3->80*80, hd2->160*160, Upsample 2 times
#         self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
#         self.hd3_UT_hd2_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)

#         # hd4->40*40, hd2->160*160, Upsample 4 times
#         self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
#         self.hd4_UT_hd2_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)

#         self.conv2d_1 = ConvModule(self.UpChannels, self.UpChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)  # 16

#         '''stage 1d'''
#         # h1->320*320, hd1->320*320, Concatenation
#         self.h1_Cat_hd1_conv = ConvModule(filters[0], self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)

#         # hd2->160*160, hd1->320*320, Upsample 2 times
#         self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
#         self.hd2_UT_hd1_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)

#         # hd3->80*80, hd1->320*320, Upsample 4 times
#         self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
#         self.hd3_UT_hd1_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)

#         # hd4->40*40, hd1->320*320, Upsample 8 times
#         self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
#         self.hd4_UT_hd1_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)

#         # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
#         self.conv1d_1 = ConvModule(self.UpChannels, self.UpChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)  # 16

#         # -------------Bilinear Upsampling--------------
#         self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear')
#         self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear')
#         self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

#         # DeepSup
#         self.outconv1 = Conv2d(self.UpChannels, self.num_classes, 3, padding=1)
#         self.outconv2 = Conv2d(self.UpChannels, self.num_classes, 3, padding=1)
#         self.outconv3 = Conv2d(self.UpChannels, self.num_classes, 3, padding=1)
#         self.outconv4 = Conv2d(self.UpChannels, self.num_classes, 3, padding=1)

#         self.cls = nn.Sequential(
#                     nn.Dropout(p=0.5),
#                     Conv2d(filters[3],self.num_classes, 1),
#                     #nn.AdaptiveMaxPool2d(1), # original
#                     nn.AdaptiveMaxPool2d(1),
#                     nn.Sigmoid())

#         ## dyhead
#         self.L_size = 4
#         self.S_size = 48
#         self.C_size = self.num_classes
#         self.dy_head = DyHead(4,L=self.L_size,S=self.S_size**2,C=self.C_size)
    
#     def dotProduct(self,seg,cls):
#         # cls : B x 11 
#         # seg : B x 11 x 128 x 128 
#         B, N, H, W = seg.size()
#         seg = seg.view(B, N, H * W)
#         final = torch.einsum("ijk,ij->ijk", [seg, cls])
#         final = final.view(B, N, H, W)
#         return final

#     def forward(self, inputs):
#         ## -------------Encoder-------------
#         x = self._transform_inputs(inputs) # transform style : multi select여야 한다.
#         h1,h2,h3,h4 = x
        
#         # -------------Classification-------------
#         ## custom
#         cls_branch = self.cls(h4).squeeze(3).squeeze(2) # (B,11)
#         cls_branch_max = (cls_branch>=0.5).type(torch.int)

#         ## -------------Decoder-------------
#         h1_PT_hd4 = self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))
#         h2_PT_hd4 = self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))
#         h3_PT_hd4 = self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))
#         h4_Cat_hd4 = self.h4_Cat_hd4_conv(h4)
#         hd4 = self.conv4d_1(
#             torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4), 1))# hd4->40*40*UpChannels

#         h1_PT_hd3 = self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))
#         h2_PT_hd3 = self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))
#         h3_Cat_hd3 = self.h3_Cat_hd3_conv(h3)
#         hd4_UT_hd3 = self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))
#         hd3 = self.conv3d_1(
#             torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3), 1))# hd3->80*80*UpChannels

#         h1_PT_hd2 = self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))
#         h2_Cat_hd2 = self.h2_Cat_hd2_conv(h2)
#         hd3_UT_hd2 = self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))
#         hd4_UT_hd2 = self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))
#         hd2 = self.conv2d_1(
#             torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2), 1)) # hd2->160*160*UpChannels

#         h1_Cat_hd1 = self.h1_Cat_hd1_conv(h1)
#         hd2_UT_hd1 = self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))
#         hd3_UT_hd1 = self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))
#         hd4_UT_hd1 = self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))
#         hd1 = self.conv1d_1(
#             torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1), 1))# hd1->320*320*UpChannels

#         d4 = self.outconv4(hd4)
#         d4 = self.upscore4(d4) # 16->128

#         d3 = self.outconv3(hd3)
#         d3 = self.upscore3(d3) # 32->128

#         d2 = self.outconv2(hd2)
#         d2 = self.upscore2(d2) # 64->128

#         d1 = self.outconv1(hd1) # 128

#         d1 = self.dotProduct(d1, cls_branch_max) # B x classes x H x W
#         d2 = self.dotProduct(d2, cls_branch_max)
#         d3 = self.dotProduct(d3, cls_branch_max)
#         d4 = self.dotProduct(d4, cls_branch_max)

#         # -------------Dyhead--------------
#         assert len(x)==self.L

#         levels=[]
#         heights = [ level.shape[2] for level in x]
#         median_height = int(np.median(heights))

#         assert median_height*median_height == self.S
        
#         for i in range(len(x)):
#             level = x[i]
#             # If level height is greater than median, then downsample with interpolate
#             if level.shape[2] > median_height:
#                 level = F.interpolate(input=level, size=(median_height, median_height),mode='nearest')
#             # If level height is less than median, then upsample
#             else:
#                 level = F.interpolate(input=level, size=(median_height, median_height), mode='nearest')

#             levels.append(level)
        
#         concat_levels = torch.stack(levels,dim=1) # B x level x C x H x W
#         concat_levels = concat_levels.flatten(start_dim=3).transpose(dim0=2, dim1=3) # B x level x (H*W) x C

#         output = self.dy_head(concat_levels) # B x level x (H*W) x C
#         output = output.transpose(dim0=2,dim1=3).view(output.shape[0],output.shape[1],self.C,median_height,median_height)# B x level x C x H x W

#         concat_output = [output[:,i,:,:,:] for i in range(output.shape[1])]
#         concat_output = torch.cat(concat_output,dim=1) # B x C x H x W

#         concat_output = self.cls_seg(concat_output) # B x 11 x H x W

#         return concat_output




#         dy_in = torch.stack((d1,d2,d3,d4),1) # B x level(4) x classes x H(48) x W(48)
#         dy_in = dy_in.flatten(start_dim=3).transpose(dim0=2, dim1=3) # B x level x (H*W) x classes
#         dy_out = self.dy_head(dy_in) # B x level x (H*W) x classes
#         dy_out = dy_out.transpose(dim0=2,dim1=3).view(dy_out.shape[0],self.L_size,self.C_size,self.S_size,self.S_size) # B x level x C x H x W

#         output_list = [dy_out[:,i,:,:,:] for i in range(dy_out.shape[1])]
#         output = torch.cat(output_list,dim=1) # B x(classes*4) x H x W
#         output = self.cls_seg(output)
#         return output



# @HEADS.register_module()
# class CustomDyUnetHead(BaseDecodeHead):
#     def __init__(self,
#                 feature_scale=4,
#                 is_deconv=True,
#                 is_batchnorm=True,
#                 **kwargs):
#         self.is_deconv = is_deconv
#         self.is_batchnorm = is_batchnorm
#         self.feature_scale = feature_scale
#         super(CustomDyUnetHead,self).__init__(**kwargs)
        
#         assert len(self.in_channels)==4
#         # resolution : [128x128, 64x64, 32x32, 16x16]
#         filters = self.in_channels # [128,256,512,1024]

#         self.CatChannels = filters[0] # 128
#         self.CatBlocks = len(filters) # 4
#         self.UpChannels = self.CatChannels * self.CatBlocks # 512

#         self.h1_PT_hd4 = MaxPool2d(8, 8, ceil_mode=True) # 128/8 x 128/8
#         self.h1_PT_hd4_conv = ConvModule(filters[0], 
#                             self.CatChannels, 
#                             3, 
#                             padding=1,
#                             conv_cfg=self.conv_cfg,
#                             norm_cfg=self.norm_cfg,
#                             act_cfg=self.act_cfg) # 128

#         # h2->160*160, hd4->40*40, Pooling 4 times
#         self.h2_PT_hd4 = MaxPool2d(4, 4, ceil_mode=True) # 64/4 x 64/4
#         self.h2_PT_hd4_conv = ConvModule(filters[1],
#                             self.CatChannels,
#                             3,
#                             padding=1,
#                             conv_cfg=self.conv_cfg,
#                             norm_cfg=self.norm_cfg,
#                             act_cfg=self.act_cfg) # 128

#         # h3->80*80, hd4->40*40, Pooling 2 times
#         self.h3_PT_hd4 = MaxPool2d(2, 2, ceil_mode=True) # 32/2 x 32/2
#         self.h3_PT_hd4_conv = ConvModule(filters[2],
#                                          self.CatChannels,
#                                           3, 
#                                           padding=1,
#                                           conv_cfg=self.conv_cfg,
#                                           norm_cfg=self.norm_cfg,
#                                           act_cfg=self.act_cfg) # 128

#         # h4->40*40, hd4->40*40, Concatenation
#         self.h4_Cat_hd4_conv = ConvModule(filters[3],
#                                             self.CatChannels, 
#                                             3,
#                                             padding=1,
#                                             conv_cfg=self.conv_cfg,
#                                             norm_cfg=self.norm_cfg,
#                                             act_cfg=self.act_cfg)

#         self.dy4d_1 = DyHead(4,L=4,S=16*16,C=self.CatChannels)
#         self.conv4d_1 = ConvModule(self.UpChannels, self.UpChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)# 16
        

#         '''stage 3d'''
#         # h1->320*320, hd3->80*80, Pooling 4 times
#         self.h1_PT_hd3 = MaxPool2d(4, 4, ceil_mode=True)
#         self.h1_PT_hd3_conv = ConvModule(filters[0], self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)
        

#         # h2->160*160, hd3->80*80, Pooling 2 times
#         self.h2_PT_hd3 = MaxPool2d(2, 2, ceil_mode=True)
#         self.h2_PT_hd3_conv = ConvModule(filters[1], self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)
        

#         # h3->80*80, hd3->80*80, Concatenation
#         self.h3_Cat_hd3_conv = ConvModule(filters[2], self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)
        

#         # hd4->40*40, hd4->80*80, Upsample 2 times
#         self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
#         self.hd4_UT_hd3_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)
        
#         self.dy3d_1 = DyHead(4,L=4,S=32*32,C=self.CatChannels)
#         self.conv3d_1 = ConvModule(self.UpChannels, self.UpChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)  # 16

#         '''stage 2d '''
#         # h1->320*320, hd2->160*160, Pooling 2 times
#         self.h1_PT_hd2 = MaxPool2d(2, 2, ceil_mode=True)
#         self.h1_PT_hd2_conv = ConvModule(filters[0], self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)

#         # h2->160*160, hd2->160*160, Concatenation
#         self.h2_Cat_hd2_conv = ConvModule(filters[1], self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)

#         # hd3->80*80, hd2->160*160, Upsample 2 times
#         self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
#         self.hd3_UT_hd2_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)

#         # hd4->40*40, hd2->160*160, Upsample 4 times
#         self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
#         self.hd4_UT_hd2_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)

#         self.dy2d_1 = DyHead(4,L=4,S=64*64,C=self.CatChannels)
#         self.conv2d_1 = ConvModule(self.UpChannels, self.UpChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)  # 16

#         '''stage 1d'''
#         # h1->320*320, hd1->320*320, Concatenation
#         self.h1_Cat_hd1_conv = ConvModule(filters[0], self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)

#         # hd2->160*160, hd1->320*320, Upsample 2 times
#         self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
#         self.hd2_UT_hd1_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)

#         # hd3->80*80, hd1->320*320, Upsample 4 times
#         self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
#         self.hd3_UT_hd1_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)

#         # hd4->40*40, hd1->320*320, Upsample 8 times
#         self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
#         self.hd4_UT_hd1_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)

#         # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
#         self.dy1d_1 = DyHead(4,L=4,S=128*128,C=self.CatChannels)
#         self.conv1d_1 = ConvModule(self.UpChannels, self.UpChannels, 3, padding=1,
#                                     conv_cfg=self.conv_cfg,
#                                     norm_cfg=self.norm_cfg,
#                                     act_cfg=self.act_cfg)  # 16

#         # -------------Bilinear Upsampling--------------
#         self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear')
#         self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear')
#         self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

#         # DeepSup
#         self.outconv1 = Conv2d(self.UpChannels, self.num_classes, 3, padding=1)
#         self.outconv2 = Conv2d(self.UpChannels, self.num_classes, 3, padding=1)
#         self.outconv3 = Conv2d(self.UpChannels, self.num_classes, 3, padding=1)
#         self.outconv4 = Conv2d(self.UpChannels, self.num_classes, 3, padding=1)

#         self.cls = nn.Sequential(
#                     nn.Dropout(p=0.5),
#                     Conv2d(filters[3],self.num_classes, 1),
#                     #nn.AdaptiveMaxPool2d(1), # original
#                     nn.AdaptiveMaxPool2d(1),
#                     nn.Sigmoid())

    
#     def dotProduct(self,seg,cls):
#         # cls : B x 11 
#         # seg : B x 11 x 128 x 128 
#         B, N, H, W = seg.size()
#         seg = seg.view(B, N, H * W)
#         final = torch.einsum("ijk,ij->ijk", [seg, cls])
#         final = final.view(B, N, H, W)
#         return final

#     def forward(self, inputs):
#         ## -------------Encoder-------------
#         x = self._transform_inputs(inputs)
#         h1,h2,h3,h4 = x

#         # -------------Classification-------------
#         ## custom
#         cls_branch = self.cls(h4).squeeze(3).squeeze(2) # (B,11)
#         cls_branch_max = (cls_branch>=0.5).type(torch.int)

#         ## -------------Decoder-------------
#         h1_PT_hd4 = self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))
#         h2_PT_hd4 = self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))
#         h3_PT_hd4 = self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))
#         h4_Cat_hd4 = self.h4_Cat_hd4_conv(h4)

#         dy_hd4 = torch.stack([h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4],dim=1) #  B x level x C x H x W
#         dy_hd4 = dy_hd4.flatten(start_dim=3).transpose(dim0=2, dim1=3) # B x level x (H*W) x C
#         dy_hd4 = self.dy4d_1(dy_hd4) # B x level x (H*W) x C
#         dy_hd4 = dy_hd4.transpose(dim0=2,dim1=3).view(dy_hd4.shape[0],dy_hd4.shape[1],self.CatChannels,16,16) # B x level x C x H x W
#         hd4_list = [dy_hd4[:,i,:,:,:] for i in range(dy_hd4.shape[1])]
#         hd4 = self.conv4d_1(torch.cat(hd4_list,dim=1)) # B x (C*4) x H x W

#         h1_PT_hd3 = self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))
#         h2_PT_hd3 = self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))
#         h3_Cat_hd3 = self.h3_Cat_hd3_conv(h3)
#         hd4_UT_hd3 = self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))

#         dy_hd3 = torch.stack([h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3],dim=1) #  B x level x C x H x W
#         dy_hd3 = dy_hd3.flatten(start_dim=3).transpose(dim0=2, dim1=3) # B x level x (H*W) x C
#         dy_hd3 = self.dy3d_1(dy_hd3) # B x level x (H*W) x C
#         dy_hd3 = dy_hd3.transpose(dim0=2,dim1=3).view(dy_hd3.shape[0],dy_hd3.shape[1],self.CatChannels,32,32) # B x level x C x H x W
#         hd3_list = [dy_hd3[:,i,:,:,:] for i in range(dy_hd3.shape[1])]
#         hd3 = self.conv3d_1(torch.cat(hd3_list,dim=1)) # B x C x H x W


#         h1_PT_hd2 = self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))
#         h2_Cat_hd2 = self.h2_Cat_hd2_conv(h2)
#         hd3_UT_hd2 = self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))
#         hd4_UT_hd2 = self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))
        
#         dy_hd2 = torch.stack([h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2],dim=1) #  B x level x C x H x W
#         dy_hd2 = dy_hd2.flatten(start_dim=3).transpose(dim0=2, dim1=3) # B x level x (H*W) x C
#         dy_hd2 = self.dy2d_1(dy_hd2) # B x level x (H*W) x C
#         dy_hd2 = dy_hd2.transpose(dim0=2,dim1=3).view(dy_hd2.shape[0],dy_hd2.shape[1],self.CatChannels,64,64) # B x level x C x H x W
#         hd2_list = [dy_hd2[:,i,:,:,:] for i in range(dy_hd2.shape[1])]
#         hd2 = self.conv2d_1(torch.cat(hd2_list,dim=1)) # B x C x H x W


#         h1_Cat_hd1 = self.h1_Cat_hd1_conv(h1)
#         hd2_UT_hd1 = self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))
#         hd3_UT_hd1 = self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))
#         hd4_UT_hd1 = self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))

#         dy_hd1 = torch.stack([h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1],dim=1) #  B x level x C x H x W
#         dy_hd1 = dy_hd1.flatten(start_dim=3).transpose(dim0=2, dim1=3) # B x level x (H*W) x C
#         dy_hd1 = self.dy1d_1(dy_hd1) # B x level x (H*W) x C
#         dy_hd1 = dy_hd1.transpose(dim0=2,dim1=3).view(dy_hd1.shape[0],dy_hd1.shape[1],self.CatChannels,128,128) # B x level x C x H x W
#         hd1_list = [dy_hd1[:,i,:,:,:] for i in range(dy_hd1.shape[1])]
#         hd1 = self.conv1d_1(torch.cat(hd1_list,dim=1)) # B x C x H x W

#         d4 = self.outconv4(hd4) # B x 11 x H x W
#         d4 = self.upscore4(d4) # 16->128

#         d3 = self.outconv3(hd3) # B x 11 x H x W
#         d3 = self.upscore3(d3) # 32->128

#         d2 = self.outconv2(hd2) # B x 11 x H x W
#         d2 = self.upscore2(d2) # 64->128

#         d1 = self.outconv1(hd1) # B x 11 x H x W

#         d1 = self.dotProduct(d1, cls_branch_max)
#         d2 = self.dotProduct(d2, cls_branch_max)
#         d3 = self.dotProduct(d3, cls_branch_max)
#         d4 = self.dotProduct(d4, cls_branch_max)

        
#         output = torch.cat((d1,d2,d3,d4),1)
#         output = self.cls_seg(output)
#         return output
