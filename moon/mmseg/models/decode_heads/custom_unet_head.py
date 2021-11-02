import torch
import torch.nn as nn
import torch.nn.functional as F
from .custom_unet.layers import unetConv2
from .custom_unet.init_weights import init_weights
import numpy as np
from mmcv.cnn import ConvModule
from mmcv.cnn import MaxPool2d
from mmcv.cnn import Conv2d

from ..builder import HEADS
from .decode_head import BaseDecodeHead


class CBAM(nn.Module):
    def __init__(self,in_channels):
        super(CBAM,self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//2,kernel_size=1),
            nn.Conv2d(in_channels//2,in_channels,kernel_size=1))
        self.sigmoid = nn.Sigmoid()

        self.spatial=nn.Sequential(
            nn.Conv2d(2,1,kernel_size=3,padding=1),
            nn.Sigmoid())

    def forward(self,input):
        ## channel attention
        max_input = self.maxpool(input) # N x C x 1 x 1
        avg_input = self.avgpool(input) # N x C x 1 x 1

        max_mlp = self.mlp(max_input) # N x C x 1 x 1
        avg_mlp = self.mlp(avg_input) # N x C x 1 x 1
        channel_attended = self.sigmoid(max_mlp + avg_mlp) * input # N x C x H x W

        ## spatial attention
        spatial_input = torch.stack([input.max(axis=1)[0],input.mean(axis=1)],axis=1) # N x 2 x H x W
        spatial_output = self.spatial(spatial_input) # N x 1 x H x W
        output = spatial_output * channel_attended
        return output


@HEADS.register_module()
class CustomUnetCBAMHead(BaseDecodeHead):
    def __init__(self,
                feature_scale=4,
                is_deconv=True,
                is_batchnorm=True,
                **kwargs):
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        super(CustomUnetCBAMHead,self).__init__(**kwargs)
        
        assert len(self.in_channels)==4
        # resolution : [128x128, 64x64, 32x32, 16x16]
        filters = self.in_channels # [128,256,512,1024]

        self.CatChannels = filters[0] # 128
        self.CatBlocks = len(filters) # 4
        self.UpChannels = self.CatChannels * self.CatBlocks # 512

        

        self.h1_PT_hd4 = MaxPool2d(8, 8, ceil_mode=True) # 128/8 x 128/8
        self.h1_PT_hd4_conv = ConvModule(filters[0], 
                            self.CatChannels, 
                            3, 
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg) # 128

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = MaxPool2d(4, 4, ceil_mode=True) # 64/4 x 64/4
        self.h2_PT_hd4_conv = ConvModule(filters[1],
                            self.CatChannels,
                            3,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg) # 128

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = MaxPool2d(2, 2, ceil_mode=True) # 32/2 x 32/2
        self.h3_PT_hd4_conv = ConvModule(filters[2],
                                         self.CatChannels,
                                          3, 
                                          padding=1,
                                          conv_cfg=self.conv_cfg,
                                          norm_cfg=self.norm_cfg,
                                          act_cfg=self.act_cfg) # 128

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = ConvModule(filters[3],
                                            self.CatChannels, 
                                            3,
                                            padding=1,
                                            conv_cfg=self.conv_cfg,
                                            norm_cfg=self.norm_cfg,
                                            act_cfg=self.act_cfg)

        self.conv4d_1 = ConvModule(self.UpChannels, self.UpChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)# 16

        self.conv4d_1_cbam = CBAM(self.UpChannels)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = ConvModule(filters[0], self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)
        

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = ConvModule(filters[1], self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)
        

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = ConvModule(filters[2], self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)
        

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)
        
        self.conv3d_1 = ConvModule(self.UpChannels, self.UpChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)  # 16
        
        self.conv3d_1_cbam = CBAM(self.UpChannels)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = ConvModule(filters[0], self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = ConvModule(filters[1], self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)

        self.conv2d_1 = ConvModule(self.UpChannels, self.UpChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)  # 16

        self.conv2d_1_cbam = CBAM(self.UpChannels)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = ConvModule(filters[0], self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)

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

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = ConvModule(self.UpChannels, self.UpChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)  # 16

        self.conv1d_1_cbam = CBAM(self.UpChannels)

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
                    #nn.AdaptiveMaxPool2d(1), # original
                    nn.AdaptiveMaxPool2d(1),
                    nn.Sigmoid())

    
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
        x = self._transform_inputs(inputs)
        h1,h2,h3,h4 = x

        # -------------Classification-------------
        ## custom
        cls_branch = self.cls(h4).squeeze(3).squeeze(2) # (B,11)
        cls_branch_max = (cls_branch>=0.5).type(torch.int)

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))
        h2_PT_hd4 = self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))
        h3_PT_hd4 = self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))
        h4_Cat_hd4 = self.h4_Cat_hd4_conv(h4)

        
        hd4_input = torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4), 1)
        hd4 = self.conv4d_1(self.conv4d_1_cbam(hd4_input))# hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))
        h2_PT_hd3 = self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))
        h3_Cat_hd3 = self.h3_Cat_hd3_conv(h3)
        hd4_UT_hd3 = self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))
        
        hd3_input = torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3), 1)
        hd3 = self.conv3d_1(self.conv3d_1_cbam(hd3_input))# hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))
        h2_Cat_hd2 = self.h2_Cat_hd2_conv(h2)
        hd3_UT_hd2 = self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))
        hd4_UT_hd2 = self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))

        hd2_input = torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2), 1)
        hd2 = self.conv2d_1(self.conv2d_1_cbam(hd2_input)) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_conv(h1)
        hd2_UT_hd1 = self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))
        hd3_UT_hd1 = self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))
        hd4_UT_hd1 = self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))

        hd1_input = torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1), 1)
        hd1 = self.conv1d_1(self.conv1d_1_cbam(hd1_input))# hd1->320*320*UpChannels

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) # 16->128

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) # 32->128

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2) # 64->128

        d1 = self.outconv1(hd1) # 128

        d1 = self.dotProduct(d1, cls_branch_max)
        d2 = self.dotProduct(d2, cls_branch_max)
        d3 = self.dotProduct(d3, cls_branch_max)
        d4 = self.dotProduct(d4, cls_branch_max)
        # d5 = self.dotProduct(d5, cls_branch_max)

        ## custom
        output = torch.cat((d1,d2,d3,d4),1)
        output = self.cls_seg(output)
        return output


@HEADS.register_module()
class CustomUnetHead(BaseDecodeHead):
    def __init__(self,
                feature_scale=4,
                is_deconv=True,
                is_batchnorm=True,
                **kwargs):
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        super(CustomUnetHead,self).__init__(**kwargs)
        
        assert len(self.in_channels)==4
        # resolution : [128x128, 64x64, 32x32, 16x16]
        filters = self.in_channels # [128,256,512,1024]

        self.CatChannels = filters[0] # 128
        self.CatBlocks = len(filters) # 4
        self.UpChannels = self.CatChannels * self.CatBlocks # 512


        self.h1_PT_hd4 = MaxPool2d(8, 8, ceil_mode=True) # 128/8 x 128/8
        self.h1_PT_hd4_conv = ConvModule(filters[0], 
                            self.CatChannels, 
                            3, 
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg) # 128

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = MaxPool2d(4, 4, ceil_mode=True) # 64/4 x 64/4
        self.h2_PT_hd4_conv = ConvModule(filters[1],
                            self.CatChannels,
                            3,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg) # 128

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = MaxPool2d(2, 2, ceil_mode=True) # 32/2 x 32/2
        self.h3_PT_hd4_conv = ConvModule(filters[2],
                                         self.CatChannels,
                                          3, 
                                          padding=1,
                                          conv_cfg=self.conv_cfg,
                                          norm_cfg=self.norm_cfg,
                                          act_cfg=self.act_cfg) # 128

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = ConvModule(filters[3],
                                            self.CatChannels, 
                                            3,
                                            padding=1,
                                            conv_cfg=self.conv_cfg,
                                            norm_cfg=self.norm_cfg,
                                            act_cfg=self.act_cfg)

        self.conv4d_1 = ConvModule(self.UpChannels, self.UpChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)# 16
        

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = ConvModule(filters[0], self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)
        

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = ConvModule(filters[1], self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)
        

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = ConvModule(filters[2], self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)
        

        # hd4->40*40, hd4->80*80, Upsample 2 times
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
        self.h1_PT_hd2_conv = ConvModule(filters[0], self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = ConvModule(filters[1], self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = ConvModule(self.UpChannels, self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)

        # hd4->40*40, hd2->160*160, Upsample 4 times
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
        self.h1_Cat_hd1_conv = ConvModule(filters[0], self.CatChannels, 3, padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)

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

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
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
                    #nn.AdaptiveMaxPool2d(1), # original
                    nn.AdaptiveMaxPool2d(1),
                    nn.Sigmoid())

    
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
        x = self._transform_inputs(inputs)
        h1,h2,h3,h4 = x

        # -------------Classification-------------
        ## custom
        cls_branch = self.cls(h4).squeeze(3).squeeze(2) # (B,11)
        cls_branch_max = (cls_branch>=0.5).type(torch.int)

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))
        h2_PT_hd4 = self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))
        h3_PT_hd4 = self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))
        h4_Cat_hd4 = self.h4_Cat_hd4_conv(h4)
        hd4 = self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4), 1))# hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))
        h2_PT_hd3 = self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))
        h3_Cat_hd3 = self.h3_Cat_hd3_conv(h3)
        hd4_UT_hd3 = self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))
        hd3 = self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3), 1))# hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))
        h2_Cat_hd2 = self.h2_Cat_hd2_conv(h2)
        hd3_UT_hd2 = self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))
        hd4_UT_hd2 = self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))
        hd2 = self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2), 1)) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_conv(h1)
        hd2_UT_hd1 = self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))
        hd3_UT_hd1 = self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))
        hd4_UT_hd1 = self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))
        hd1 = self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1), 1))# hd1->320*320*UpChannels

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) # 16->128

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) # 32->128

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2) # 64->128

        d1 = self.outconv1(hd1) # 128

        d1 = self.dotProduct(d1, cls_branch_max)
        d2 = self.dotProduct(d2, cls_branch_max)
        d3 = self.dotProduct(d3, cls_branch_max)
        d4 = self.dotProduct(d4, cls_branch_max)
        # d5 = self.dotProduct(d5, cls_branch_max)

        ## custom
        output = torch.cat((d1,d2,d3,d4),1)
        output = self.cls_seg(output)
        return output
