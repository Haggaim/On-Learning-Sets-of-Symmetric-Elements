# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# -----
# The following software may be included in this product: pytorch-unet,
# https://github.com/usuyama/pytorch-unet
# License: https://github.com/usuyama/pytorch-unet/blob/master/LICENSE
# MIT License
#
# Copyright (c) 2018 Naoto Usuyama
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F

from deblurring_exp.layers import \
    SetDropout,ReluSets,Conv2dDeepSym,Conv2dSiamese,\
    SetMaxPool2d,SetUpsample,DeepSetsBlock,Conv2dSAittala,Conv2dSridhar


def double_conv(in_channels, out_channels,model_type,p_drop,use_max):
    if model_type == 'deepsets':
        return nn.Sequential(
            Conv2dSiamese(in_channels, out_channels, 3,padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop),
            Conv2dSiamese(out_channels, out_channels, 3,padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop)
        )
    elif model_type == 'Sridhar':
        return nn.Sequential(
            Conv2dSridhar(in_channels, out_channels, 3, padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop),
            Conv2dSridhar(out_channels, out_channels, 3, padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop)
        )
    elif model_type == 'Aittala':
        return nn.Sequential(
            Conv2dSAittala(in_channels, out_channels, 3, padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop),
            Conv2dSAittala(out_channels, out_channels, 3, padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop)
        )
    else:
        return nn.Sequential(
            Conv2dDeepSym(in_channels, out_channels, 3,padding=1,use_max=use_max),
            ReluSets(),
            SetDropout(p_drop=p_drop),
            Conv2dDeepSym(out_channels, out_channels, 3,padding=1,use_max=use_max),
            ReluSets(),
            SetDropout(p_drop=p_drop)
        )


def get_feature_processing_block(model_type,channels):
    if model_type == 'deepsets' or model_type == 'DeepSymmetricNet' or\
            model_type == 'Aittala' or model_type == 'Sridhar':
        return DeepSetsBlock(channels=(channels[5],channels[5],channels[5]))


class UNetLarge(nn.Module):
    def __init__(self,model_type,p_drop,use_max):
        super().__init__()
        self.model_type = model_type

        if self.model_type =='deepsets' or self.model_type == 'Sridhar':
            c = (3, 48, 64, 128, 200, 300)
        elif self.model_type == 'DeepSymmetricNet':
            c = (3, 48, 50, 100, 150, 200)
        elif model_type == 'Aittala':
            c = (3, 150, 200, 220, 250, 250)

        self.dconv_down1 = double_conv(c[0], c[1],self.model_type,p_drop,use_max=use_max)
        self.dconv_down2 = double_conv(c[1], c[2],self.model_type,p_drop,use_max=use_max)
        self.dconv_down3 = double_conv(c[2], c[3],self.model_type,p_drop,use_max=use_max)
        self.dconv_down4 = double_conv(c[3], c[4],self.model_type,p_drop,use_max=use_max)
        self.dconv_down5 = double_conv(c[4], c[5],self.model_type,p_drop,use_max=use_max)

        self.maxpool2 = SetMaxPool2d(stride=2)
        self.maxpool8 = SetMaxPool2d(stride=8)

        self.feature_processing_block = get_feature_processing_block(model_type=model_type,channels=c)

        self.upsample2 = SetUpsample(scale_factor=2)
        self.upsample8 = SetUpsample(scale_factor=8)

        self.dconv_up4 = double_conv(c[5] + c[5], c[4],self.model_type,p_drop,use_max=use_max)
        self.dconv_up3 = double_conv(c[4] + c[4], c[3],self.model_type,p_drop,use_max=use_max)
        self.dconv_up2 = double_conv(c[3] + c[3], c[2],self.model_type,p_drop,use_max=use_max)
        self.dconv_up1 = double_conv(c[2] + c[2], c[1],self.model_type,p_drop,use_max=use_max)

        self.conv1 = nn.Conv2d(c[1] + c[1], c[1],kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c[1], 3,kernel_size=3, padding=1)



    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool2(conv1) #64

        conv2 = self.dconv_down2(x)
        x = self.maxpool2(conv2) #32

        conv3 = self.dconv_down3(x)
        x = self.maxpool2(conv3) #16

        conv4 = self.dconv_down4(x)
        x = self.maxpool2(conv4)#8

        conv5 = self.dconv_down5(x)
        x = self.maxpool8(conv5)

        # deep sets block
        x = x.squeeze()
        x = self.feature_processing_block(x)
        x = x.unsqueeze(dim=3).unsqueeze(dim=4)
        x = self.upsample8(x)#8
        x = torch.cat([x, conv5], dim=2)

        x = self.dconv_up4(x)
        x = self.upsample2(x)#16
        x = torch.cat([x, conv4], dim=2)

        x = self.dconv_up3(x)
        x = self.upsample2(x)#32
        x = torch.cat([x, conv3], dim=2)

        x = self.dconv_up2(x)
        x = self.upsample2(x)#64
        x = torch.cat([x, conv2], dim=2)

        x = self.dconv_up1(x)
        x = self.upsample2(x)  # 128
        x = torch.cat([x, conv1], dim=2)
        x = x.max(dim=1,keepdim=False)[0]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = torch.clamp(x,-1,1)

        return x

