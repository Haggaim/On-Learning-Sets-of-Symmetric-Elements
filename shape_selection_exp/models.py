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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb
from shape_selection_exp.layers import Conv1dBlock, DeepSetsBlock, DeepSymmetricConv1dBlock, SridharConvBlock, AittalaConvBlock
from shape_selection_exp.graph_layers import GCN1dBlock, DeepSymmetricGCN1dBlock

class DeepSetsNet(nn.Module):
    def __init__(self,args):
        super(DeepSetsNet, self).__init__()        
        self.use_graph = args.use_graph
        # layers        
        if self.use_graph:
            self.conv_block_1 = GCN1dBlock([3, 64, 128*2, 128], args)
        else:
            self.conv_block_1 = Conv1dBlock([3, 64, 128*2, 128], args)
        self.deep_sets_block = DeepSetsBlock(args, channels=(128, 128, 1))

    def forward(self, x):
        
        if self.use_graph:
            b,n,c,l = x['x'].size()            
            x = self.conv_block_1(x['x'], x['edge_index'])  # (b*n, 1024, 6890)
            
        else:
            b,n,c,l = x.size()
            x = self.conv_block_1(x)            
                
        x = torch.max(x, 2, keepdim=False)[0]  # (b*n, 1024)        
        x = x.view(b,n,128)
        x = self.deep_sets_block(x)

        return F.log_softmax(x, dim=1)


class DeepSymmetricNet(nn.Module):

    def __init__(self, args):
        super(DeepSymmetricNet, self).__init__()
        self.use_graph = args.use_graph
        # layers
        self.dim_factor = args.dim_factor
        if self.use_graph:
            self.conv_block_1 = DeepSymmetricGCN1dBlock([3, 64, 128*self.dim_factor, 128], args)
        else:     
            self.conv_block_1 = DeepSymmetricConv1dBlock([3, 64, 128*self.dim_factor, 128], args)
        self.deep_sets_block = DeepSetsBlock(args, channels=(128, 128, 1))


    def forward(self, x):
        if self.use_graph:
            b,n,c,l = x['x'].size()            
            x = self.conv_block_1(x['x'], x['edge_index'])  # (b*n, 1024, 6890)
        else:
            b,n,c,l = x.size()
            x = self.conv_block_1(x)  # (b*n, 1024, 6890)
        x = torch.max(x, 2, keepdim=False)[0]  # (b*n, 1024)        
        x = x.view(b,n,128)
       
        x = self.deep_sets_block(x)

        return F.log_softmax(x, dim=1)

    
class Sridhar(nn.Module):

    def __init__(self, args):
        super(Sridhar, self).__init__()

        # layers
        self.conv_block_1 = SridharConvBlock([3, 64, 128*2, 128], args)
        self.deep_sets_block = DeepSetsBlock(args, channels=(128, 128, 1))

    def forward(self, x):
        b,n,c,l = x.size()
        x = self.conv_block_1(x)  # (b*n, 1024, 6890)
        x = torch.max(x, 2, keepdim=False)[0]  # (b*n, 1024)        
        x = x.view(b,n,128)
       
        x = self.deep_sets_block(x)

        return F.log_softmax(x, dim=1)


class Aittala(nn.Module):

    def __init__(self, args):
        super(Aittala, self).__init__()

        # layers
        self.conv_block_1 = AittalaConvBlock([3, 64, 128*2, 64], args)
        self.deep_sets_block = DeepSetsBlock(args, channels=(128, 128, 1))

    def forward(self, x):
        b, n, c, l = x.size()
        x = self.conv_block_1(x)   
        x = torch.max(x, 2, keepdim=False)[0] 
        x = x.view(b,n,128)
        x = self.deep_sets_block(x)

        return F.log_softmax(x, dim=1)
