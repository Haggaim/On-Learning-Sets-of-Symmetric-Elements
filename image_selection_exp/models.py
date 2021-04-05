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


from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import math
from image_selection_exp.layers import ConvBlock,\
    DeepSetsBlock,DeepSymmetricConvBlock,SridharConvBlock,AittalaConvBlock


class DeepSetsNet(nn.Module):
    def __init__(self,args):
        super(DeepSetsNet, self).__init__()
        # layers
        self.conv_block_1 = ConvBlock([3, 50, 100], args)
        self.conv_block_2 = ConvBlock([100, 100, 180], args)
        self.conv_block_3 = ConvBlock([180, 200, 256], args)
        self.deep_sets_block = DeepSetsBlock(args)

    def forward(self, x):

        b,n,c,h,w = x.size()
        x = self.conv_block_1(x)
        x = F.max_pool2d(x,kernel_size = 2,stride=2)
        x = x.view(b,n,100,math.floor((h-6)/2),math.floor((w-6)/2))

        b, n, c, h, w = x.size()
        x = self.conv_block_2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(b,n,180,math.floor((h-6)/2),math.floor((w-6)/2))

        b, n, c, h, w = x.size()
        x = self.conv_block_3(x)
        x = F.max_pool2d(x, kernel_size=5, stride=5)
        x = x.view(b,n,256)

        x = self.deep_sets_block(x)

        return F.log_softmax(x, dim=1)


class DeepSymmetricNet(nn.Module):

    def __init__(self, args):
        super(DeepSymmetricNet, self).__init__()

        # layers
        self.conv_block_1 = DeepSymmetricConvBlock([3, 32, 64], args)
        self.conv_block_2 = DeepSymmetricConvBlock([64, 64, 128], args)
        self.conv_block_3 = DeepSymmetricConvBlock([128,128,256],args)
        self.deep_sets_block = DeepSetsBlock(args)

    def forward(self, x):
        b, n, c, h, w = x.size()
        x = self.conv_block_1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(b, n, 64, math.floor((h - 6) / 2), math.floor((w - 6) / 2))

        b, n, c, h, w = x.size()
        x = self.conv_block_2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(b, n, 128, math.floor((h - 6) / 2), math.floor((w - 6) / 2))

        b, n, c, h, w = x.size()
        x = self.conv_block_3(x)
        x = F.max_pool2d(x, kernel_size=5, stride=5)
        x = x.view(b, n, 256)

        x = self.deep_sets_block(x)

        return F.log_softmax(x, dim=1)


class Sridhar(nn.Module):
    def __init__(self, args):
        super(Sridhar, self).__init__()

        # layers
        self.conv_block_1 = SridharConvBlock([3, 50, 100], args)
        self.conv_block_2 = SridharConvBlock([100, 100, 180], args)
        self.conv_block_3 = SridharConvBlock([180, 200, 256],args)
        self.deep_sets_block = DeepSetsBlock(args)

    def forward(self, x):
        b, n, c, h, w = x.size()
        x = self.conv_block_1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(b, n, 100, math.floor((h - 6) / 2), math.floor((w - 6) / 2))

        b, n, c, h, w = x.size()
        x = self.conv_block_2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(b, n, 180, math.floor((h - 6) / 2), math.floor((w - 6) / 2))

        b, n, c, h, w = x.size()
        x = self.conv_block_3(x)
        x = F.max_pool2d(x, kernel_size=5, stride=5)
        x = x.view(b, n, 256)

        x = self.deep_sets_block(x)

        return F.log_softmax(x, dim=1)


class Aittala(nn.Module):

    def __init__(self, args):
        super(Aittala, self).__init__()

        # layers
        self.conv_block_1 = AittalaConvBlock([3, 90, 100], args)
        self.conv_block_2 = AittalaConvBlock([200, 100, 100], args)
        self.conv_block_3 = AittalaConvBlock([200, 110, 128],args)
        self.deep_sets_block = DeepSetsBlock(args)

    def forward(self, x):
        b, n, c, h, w = x.size()
        x = self.conv_block_1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(b, n, 200, math.floor((h - 6) / 2), math.floor((w - 6) / 2))

        b, n, c, h, w = x.size()
        x = self.conv_block_2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(b, n, 200, math.floor((h - 6) / 2), math.floor((w - 6) / 2))

        b, n, c, h, w = x.size()
        x = self.conv_block_3(x)
        x = F.max_pool2d(x, kernel_size=5, stride=5)
        x = x.view(b, n, 256)

        x = self.deep_sets_block(x)

        return F.log_softmax(x, dim=1)