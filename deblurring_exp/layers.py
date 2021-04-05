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


import torch
import torch.nn as nn
import torch.nn.functional as F


class SetDropout(nn.Module):
    def __init__(self,p_drop):
        super().__init__()
        self.drop = nn.Dropout2d(p=p_drop)
    def forward(self, x):
        b,n,c,h,w = x.size()
        x = x.view(b*n,c,h,w)
        x = self.drop(x)
        x = x.view(b, n, c, h, w)
        return x


class Conv2dSAittala(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size,padding=1):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, int(out_channels/2), kernel_size, padding=padding)
        torch.nn.init.xavier_normal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(num_features=out_channels)


    def forward(self, x):
        b, n, c, h, w = x.size()
        x = self.conv(x.view(n * b, c, h, w))
        x = x.view(b, n, int(self.out_channels/2), h, w)
        x_max = x.max(dim=1, keepdim=True)[0]
        x = torch.cat([x, x_max.repeat(1,n,1,1,1)],dim=2)
        x = x.view(b * n, self.out_channels, h, w)
        x = self.bn(x)
        x = x.view(b, n, self.out_channels, h, w)
        return x

class Conv2dSridhar(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size,padding=1):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        torch.nn.init.xavier_normal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(num_features=out_channels)


    def forward(self, x):
        b, n, c, h, w = x.size()
        x = self.conv(x.view(n * b, c, h, w))
        x = x.view(b, n, self.out_channels, h, w)
        x_mean = x.mean(dim=1, keepdim=True)
        x = x - x_mean
        x = x.view(b * n, self.out_channels, h, w)
        x = self.bn(x)
        x = x.view(b, n, self.out_channels, h, w)
        return x

class Conv2dSiamese(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size,padding=1):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

        torch.nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        b,n,c,h,w = x.size()
        x = x.view(b*n,c,h,w)
        x = self.bn(self.conv(x))
        x = x.view(b, n, self.out_channels, h, w)
        return x


class Conv2dDeepSym(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size,padding=0,use_max=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_max = use_max
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.bns = nn.BatchNorm2d(num_features=out_channels)
        torch.nn.init.xavier_normal_(self.conv.weight)
        torch.nn.init.xavier_normal_(self.conv_s.weight)

    def forward(self, x):
        b, n, c, h, w = x.size()
        x1 = self.bn(self.conv(x.view(n * b, c, h, w)))
        if self.use_max:
            x2 = self.bns(self.conv_s(torch.max(x, dim=1, keepdim=False)[0]))
        else:
            x2 = self.bns(self.conv_s(torch.sum(x, dim=1, keepdim=False)))
        x2 = x2.view(b, 1, h, w, self.out_channels).repeat(1, n, 1, 1, 1).view(b * n, self.out_channels, h, w)
        x = x1 + x2
        x = x.view(b, n, self.out_channels, h, w)
        return x

class ReluSets(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b,n,c,h,w = x.size()
        x = x.view(b*n,c,h,w)
        x = F.relu(x)
        x = x.view(b, n, c, h, w)
        return x

class SetMaxPool2d(nn.Module):
    def __init__(self,stride):
        self.stride = stride
        super().__init__()

    def forward(self, x):
        b,n,c,h,w = x.size()
        x = x.view(b*n,c,h,w)
        x = F.max_pool2d(x,kernel_size = 2,stride=self.stride)
        x = x.view(b, n, c, int(h/self.stride), int(w/self.stride))
        return x

class SetUpsample(nn.Module):
    def __init__(self,scale_factor):
        self.scale_factor = scale_factor
        super().__init__()

    def forward(self, x):
        b, n, c, h, w = x.size()
        x = x.view(b * n, c, h, w)
        x = F.upsample(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        x = x.view(b, n, c, self.scale_factor*h, self.scale_factor*w)
        return x

class DeepSetsBlock(nn.Module):
    def __init__(self,channels=(256, 128, 1)):
        super(DeepSetsBlock, self).__init__()
        # layers
        self.channels = channels
        self.fc_1 = torch.nn.Linear(in_features=self.channels[0], out_features=self.channels[0])
        self.fc_2 = torch.nn.Linear(in_features=self.channels[0], out_features=self.channels[1])
        self.fc_3 = torch.nn.Linear(in_features=self.channels[1], out_features=self.channels[2])
        self.fc_1s = torch.nn.Linear(in_features=self.channels[0], out_features=self.channels[0])
        self.fc_2s = torch.nn.Linear(in_features=self.channels[0], out_features=self.channels[1])
        self.fc_3s = torch.nn.Linear(in_features=self.channels[1], out_features=self.channels[2])

        self.bn1 = torch.nn.BatchNorm1d(channels[0])
        self.bn2 = torch.nn.BatchNorm1d(channels[1])
        self.bn3 = torch.nn.BatchNorm1d(channels[2])

        self.bn1s = torch.nn.BatchNorm1d(channels[0])
        self.bn2s = torch.nn.BatchNorm1d(channels[1])
        self.bn3s = torch.nn.BatchNorm1d(channels[2])

        # initializations
        torch.nn.init.xavier_normal_(self.fc_1.weight)
        torch.nn.init.xavier_normal_(self.fc_2.weight)
        torch.nn.init.xavier_normal_(self.fc_3.weight)
        torch.nn.init.xavier_normal_(self.fc_1s.weight)
        torch.nn.init.xavier_normal_(self.fc_2s.weight)
        torch.nn.init.xavier_normal_(self.fc_3s.weight)

    def forward(self, x):
        b, n, c = x.size()
        x_col = x.view(b * n, c)
        x1 = self.bn1(self.fc_1(x_col))
        x2 = self.fc_1s(torch.max(x, dim=1, keepdim=False)[0])
        x2 = self.bn1s(x2.view(b,1,self.channels[0]).repeat(1,n,1).view(b*n,self.channels[0]))
        x = x1 + x2
        x = F.relu(x)
        x = x.view(b,n,self.channels[0])

        x_col = x.view(b * n, self.channels[0])
        x1 = self.bn2(self.fc_2(x_col))
        x2 = self.fc_2s(torch.max(x, dim=1, keepdim=False)[0])
        x2 = self.bn2s(x2.view(b, 1, self.channels[1]).repeat(1, n, 1).view(b * n, self.channels[1]))
        x = x1 + x2
        x = F.relu(x)
        x = x.view(b, n, self.channels[1])

        x_col = x.view(b * n, self.channels[1])
        x1 = self.bn3(self.fc_3(x_col))
        x2 = self.fc_3s(torch.max(x, dim=1, keepdim=False)[0])
        x2 = self.bn3s(x2.view(b, 1, self.channels[1]).repeat(1, n, 1).view(b * n, self.channels[2]))
        x = x1 + x2
        x = F.relu(x)
        x = x.view(b, n, self.channels[2])
        return x


class DeepSetsBlockSiamese(nn.Module):
    def __init__(self, channels=(256, 128, 1)):
        super(DeepSetsBlockSiamese, self).__init__()
        # layers
        self.channels = channels
        self.fc_1 = torch.nn.Linear(in_features=self.channels[0], out_features=self.channels[0])
        self.fc_2 = torch.nn.Linear(in_features=self.channels[0], out_features=self.channels[1])
        self.fc_3 = torch.nn.Linear(in_features=self.channels[1], out_features=self.channels[2])

        self.bn1 = torch.nn.BatchNorm1d(channels[0])
        self.bn2 = torch.nn.BatchNorm1d(channels[1])
        self.bn3 = torch.nn.BatchNorm1d(channels[2])

        # initializations
        torch.nn.init.xavier_normal_(self.fc_1.weight)
        torch.nn.init.xavier_normal_(self.fc_2.weight)
        torch.nn.init.xavier_normal_(self.fc_3.weight)

    def forward(self, x):
        b, n, c = x.size()
        x_col = x.view(b * n, c)
        x = self.bn1(self.fc_1(x_col))
        x = F.relu(x)
        x = x.view(b, n, self.channels[0])

        x_col = x.view(b * n, self.channels[0])
        x = self.bn2(self.fc_2(x_col))
        x = F.relu(x)
        x = x.view(b, n, self.channels[1])

        x_col = x.view(b * n, self.channels[1])
        x = self.bn3(self.fc_3(x_col))
        x = F.relu(x)
        x = x.view(b, n, self.channels[2])
        return x


