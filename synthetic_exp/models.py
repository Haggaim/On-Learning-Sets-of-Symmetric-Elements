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
import math

n = 25
l = 100
num_labels = 3

class MLPNet(nn.Module):
    def __init__(self,args):
        super(MLPNet, self).__init__()
        #layers
        self.n1 = 840
        self.n2 = 420

        self.fc1 = nn.Linear(l*n, self.n1)
        self.fc2 = nn.Linear(self.n1, self.n1)
        self.fc3 = nn.Linear(self.n1, self.n2)
        self.fc4 = nn.Linear(self.n2, num_labels)
        self.bn1 = torch.nn.BatchNorm1d(self.n1)
        self.bn2 = torch.nn.BatchNorm1d(self.n1)
        self.bn3 = torch.nn.BatchNorm1d(self.n2)

        # initializations
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)



    def forward(self, x):
        x = x.view(-1,l*n)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)



class DeepSetsNet(nn.Module):
    def __init__(self,args):
        super(DeepSetsNet, self).__init__()
        #layers
        self.n1 = 1000
        self.n2 = 500
        self.fc1 = nn.Linear(l,self.n1)
        self.fc2 = nn.Linear(self.n1, self.n1)
        self.fc3 = nn.Linear(self.n1, self.n2)
        self.fc1s = nn.Linear(l, self.n1)
        self.fc2s = nn.Linear(self.n1, self.n1)
        self.fc3s = nn.Linear(self.n1, self.n2)
        self.fc4 = nn.Linear(self.n2, num_labels)
        self.bn1 = torch.nn.BatchNorm1d(self.n1)
        self.bn2 = torch.nn.BatchNorm1d(self.n1)
        self.bn3 = torch.nn.BatchNorm1d(self.n2)
        self.bn1s = torch.nn.BatchNorm1d(self.n1)
        self.bn2s = torch.nn.BatchNorm1d(self.n1)
        self.bn3s = torch.nn.BatchNorm1d(self.n2)

        # initializations
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc1s.weight)
        torch.nn.init.xavier_uniform_(self.fc2s.weight)
        torch.nn.init.xavier_uniform_(self.fc3s.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)



    def forward(self, x):
        b = len(x)
        n = len(x[0])

        x = x.view(n*b,l)
        x1 = self.bn1(self.fc1(x))
        x2 = self.bn1s(self.fc1s(x)).view(b,n,self.n1)
        x = F.relu(x1.view(b,n,self.n1) + torch.sum(x2,dim=1).view(-1,1,self.n1).repeat(1,n,1))

        x = x.view(n * b, self.n1)
        x1 = self.bn2(self.fc2(x))
        x2 = self.bn2s(self.fc2s(x)).view(b,n,self.n1)
        x = F.relu(x1.view(b,n,self.n1) + torch.sum(x2, dim=1).view(-1, 1, self.n1).repeat(1, n, 1))

        x = x.view(n * b, self.n1)
        x1 = self.bn3(self.fc3(x))
        x2 = self.bn3s(self.fc3s(x)).view(b,n,self.n2)
        x = F.relu(x1.view(b,n,self.n2) + torch.sum(x2, dim=1).view(-1, 1, self.n2).repeat(1, n, 1))

        x = torch.sum(x,dim=1)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class DeepSetsNetMax(nn.Module):
    def __init__(self,args):
        super(DeepSetsNetMax, self).__init__()
        #layers
        self.n1 = 1000
        self.n2 = 500
        self.fc1 = nn.Linear(l,self.n1)
        self.fc2 = nn.Linear(self.n1, self.n1)
        self.fc3 = nn.Linear(self.n1, self.n2)
        self.fc1s = nn.Linear(l, self.n1)
        self.fc2s = nn.Linear(self.n1, self.n1)
        self.fc3s = nn.Linear(self.n1, self.n2)
        self.fc4 = nn.Linear(self.n2, num_labels)
        self.bn1 = torch.nn.BatchNorm1d(self.n1)
        self.bn2 = torch.nn.BatchNorm1d(self.n1)
        self.bn3 = torch.nn.BatchNorm1d(self.n2)
        self.bn1s = torch.nn.BatchNorm1d(self.n1)
        self.bn2s = torch.nn.BatchNorm1d(self.n1)
        self.bn3s = torch.nn.BatchNorm1d(self.n2)

        # initializations
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc1s.weight)
        torch.nn.init.xavier_uniform_(self.fc2s.weight)
        torch.nn.init.xavier_uniform_(self.fc3s.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)



    def forward(self, x):
        b = len(x)
        n = len(x[0])

        x = x.view(n*b,l)
        x1 = self.bn1(self.fc1(x))
        x2 = self.bn1s(self.fc1s(x)).view(b,n,self.n1)
        x = F.relu(x1.view(b,n,self.n1) + torch.max(x2,dim=1)[0].view(-1,1,self.n1).repeat(1,n,1))

        x = x.view(n * b, self.n1)
        x1 = self.bn2(self.fc2(x))
        x2 = self.bn2s(self.fc2s(x)).view(b,n,self.n1)
        x = F.relu(x1.view(b,n,self.n1) + torch.max(x2, dim=1)[0].view(-1, 1, self.n1).repeat(1, n, 1))

        x = x.view(n * b, self.n1)
        x1 = self.bn3(self.fc3(x))
        x2 = self.bn3s(self.fc3s(x)).view(b,n,self.n2)
        x = F.relu(x1.view(b,n,self.n2) + torch.max(x2, dim=1)[0].view(-1, 1, self.n2).repeat(1, n, 1))

        x = torch.sum(x,dim=1)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class DeepSetsNetMax2(nn.Module):
    def __init__(self,args):
        super(DeepSetsNetMax2, self).__init__()
        #layers
        self.n1 = 1000
        self.n2 = 500
        self.fc1 = nn.Linear(l,self.n1)
        self.fc2 = nn.Linear(self.n1, self.n1)
        self.fc3 = nn.Linear(self.n1, self.n2)
        self.fc1s = nn.Linear(l, self.n1)
        self.fc2s = nn.Linear(self.n1, self.n1)
        self.fc3s = nn.Linear(self.n1, self.n2)
        self.fc4 = nn.Linear(self.n2, num_labels)
        self.bn1 = torch.nn.BatchNorm1d(self.n1)
        self.bn2 = torch.nn.BatchNorm1d(self.n1)
        self.bn3 = torch.nn.BatchNorm1d(self.n2)
        self.bn1s = torch.nn.BatchNorm1d(self.n1)
        self.bn2s = torch.nn.BatchNorm1d(self.n1)
        self.bn3s = torch.nn.BatchNorm1d(self.n2)

        # initializations
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc1s.weight)
        torch.nn.init.xavier_uniform_(self.fc2s.weight)
        torch.nn.init.xavier_uniform_(self.fc3s.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)



    def forward(self, x):
        b = len(x)
        n = len(x[0])

        x_flat = x.view(n*b,l)
        x1 = self.bn1(self.fc1(x_flat))
        x2 = self.bn1s(self.fc1s(x.max(dim=1)[0])).view(b,1,self.n1).repeat(1,n,1)
        x = F.relu(x1.view(b,n,self.n1) + x2)

        x_flat = x.view(n * b, self.n1)
        x1 = self.bn2(self.fc2(x_flat))
        x2 = self.bn2s(self.fc2s(x.max(dim=1)[0])).view(b, 1, self.n1).repeat(1, n, 1)
        x = F.relu(x1.view(b, n, self.n1) + x2)

        x_flat = x.view(n * b, self.n1)
        x1 = self.bn3(self.fc3(x_flat))
        x2 = self.bn3s(self.fc3s(x.max(dim=1)[0])).view(b, 1, self.n2).repeat(1, n, 1)
        x = F.relu(x1.view(b, n, self.n2) + x2)

        x = torch.sum(x,dim=1)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class SiameseNet(nn.Module):
    def __init__(self,args):
        super(SiameseNet, self).__init__()
        # make sure that the number of parameters is roughly the same as in the MLPs

        self.n1 = 220
        self.n2 = 110
        # layers
        self.conv1 = nn.Conv1d(1, self.n1, kernel_size=l,stride=2, padding=l-1, padding_mode='circular')
        self.conv2 = nn.Conv1d(self.n1, self.n1, kernel_size=int(l/2),stride=2, padding=int(l/2)-1, padding_mode='circular')
        self.conv3 = nn.Conv1d(self.n1, self.n2, kernel_size=int(l/4),stride=2, padding=int(l/4)-1, padding_mode='circular')
        self.fc1 = nn.Linear(self.n2,num_labels)
        self.bn1 = torch.nn.BatchNorm1d(self.n1)
        self.bn2 = torch.nn.BatchNorm1d(self.n1)
        self.bn3 = torch.nn.BatchNorm1d(self.n2)
        # initializations
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        b = len(x)
        n = len(x[1])
        # siamese part
        x = x.view(n*b,1,l)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # reshape to sets form again
        x = x.view(b,n,self.n2,math.ceil(l/8))
        x = torch.sum(torch.sum(x,dim=3),dim=1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

class SiameseDSNet(nn.Module):
    def __init__(self,args):
        super(SiameseDSNet, self).__init__()
        self.n1 = 200
        self.n2 = 100
        # layers
        self.conv1 = nn.Conv1d(1, self.n1, kernel_size=l,stride=2, padding=l-1, padding_mode='circular')
        self.conv2 = nn.Conv1d(self.n1, self.n1, kernel_size=int(l/2),stride=2, padding=int(l/2)-1, padding_mode='circular')
        self.fc3 = nn.Linear(self.n1*int(l/4), self.n2)
        self.fc3s = nn.Linear(self.n1*int(l/4), self.n2)

        self.fc1 = nn.Linear(self.n2,num_labels)
        self.bn1 = torch.nn.BatchNorm1d(self.n1)
        self.bn2 = torch.nn.BatchNorm1d(self.n1)
        self.bn3s = torch.nn.BatchNorm1d(self.n2)
        self.bn3 = torch.nn.BatchNorm1d(self.n2)

        # initializations
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc3s.weight)


    def forward(self, x):
        b = len(x)
        n = len(x[1])
        # siamese part
        x = x.view(n*b,1,l)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x_flat = x.view(n * b, int(l/4)*self.n1)
        x_flat2 = x.view(b,n, int(l / 4) * self.n1)

        x1 = self.bn3(self.fc3(x_flat))
        x2 = self.bn3s(self.fc3s(x_flat2.max(dim=1)[0])).view(b, 1, self.n2).repeat(1, n, 1)
        x = F.relu(x1.view(b, n, self.n2) + x2)

        x = torch.sum(x,dim=1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class Sridhar(nn.Module):
    def __init__(self,args):
        super(Sridhar, self).__init__()
        # make sure that the number of parameters is roughly the same as in the MLPs
        self.n1 = 220
        self.n2 = 110
        # layers
        self.conv1 = nn.Conv1d(1, self.n1, kernel_size=l,stride=2, padding=l-1, padding_mode='circular')
        self.conv2 = nn.Conv1d(self.n1, self.n1, kernel_size=int(l/2),stride=2, padding=int(l/2)-1, padding_mode='circular')
        self.conv3 = nn.Conv1d(self.n1, self.n2, kernel_size=int(l/4),stride=2, padding=int(l/4)-1, padding_mode='circular')
        self.bn1 = torch.nn.BatchNorm1d(self.n1)
        self.bn2 = torch.nn.BatchNorm1d(self.n1)
        self.bn3 = torch.nn.BatchNorm1d(self.n2)
        self.fc1 = nn.Linear(self.n2,num_labels)

        # initializations
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        b = len(x)

        x = x.view(n*b,1,l)
        cl = math.ceil(l/2)
        x1 = self.bn1(self.conv1(x)).view(b,n,self.n1,cl)
        x2 = torch.mean(x1, dim=1, keepdim=True).repeat(1, n, 1, 1)
        x = F.relu(x1 - x2)

        x = x.view(n * b, self.n1, cl)
        cl = math.ceil(cl/2)
        x1 = self.bn2(self.conv2(x)).view(b, n, self.n1, cl)
        x2 = torch.mean(x1, dim=1, keepdim=True).repeat(1, n, 1, 1)
        x = F.relu(x1 - x2)

        x = x.view(n * b, self.n1, cl)
        cl = math.ceil(cl/2)
        x1 = self.bn3(self.conv3(x)).view(b, n, self.n2, cl)
        x2 = torch.mean(x1, dim=1, keepdim=True).repeat(1, n, 1, 1)
        x = F.relu(x1 - x2)

        x = torch.sum(torch.sum(x,dim=3),dim=1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class DeepSymmetricNet(nn.Module):
    def __init__(self,args):
        super(DeepSymmetricNet, self).__init__()
        # make sure that the number of parameters is roughly the same as in the MLPs
        self.n1 = 160
        self.n2 = 80
        # layers
        self.conv1 = nn.Conv1d(1, self.n1, kernel_size=l,stride=2, padding=l-1, padding_mode='circular')
        self.conv2 = nn.Conv1d(self.n1, self.n1, kernel_size=int(l/2),stride=2, padding=int(l/2)-1, padding_mode='circular')
        self.conv3 = nn.Conv1d(self.n1, self.n2, kernel_size=int(l/4),stride=2, padding=int(l/4)-1, padding_mode='circular')
        self.conv1s = nn.Conv1d(1, self.n1, kernel_size=l,stride=2, padding=l-1, padding_mode='circular')
        self.conv2s = nn.Conv1d(self.n1, self.n1, kernel_size=int(l/2),stride=2, padding=int(l/2)-1, padding_mode='circular')
        self.conv3s = nn.Conv1d(self.n1, self.n2, kernel_size=int(l/4),stride=2, padding=int(l/4)-1, padding_mode='circular')
        self.bn1 = torch.nn.BatchNorm1d(self.n1)
        self.bn2 = torch.nn.BatchNorm1d(self.n1)
        self.bn3 = torch.nn.BatchNorm1d(self.n2)
        self.bn1s = torch.nn.BatchNorm1d(self.n1)
        self.bn2s = torch.nn.BatchNorm1d(self.n1)
        self.bn3s = torch.nn.BatchNorm1d(self.n2)
        self.fc1 = nn.Linear(self.n2,num_labels)

        # initializations
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv1s.weight)
        torch.nn.init.xavier_uniform_(self.conv2s.weight)
        torch.nn.init.xavier_uniform_(self.conv3s.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        b = len(x)
        x = x.view(n*b,1,l)
        cl = math.ceil(l/2)
        x1 = self.bn1(self.conv1(x)).view(b,n,self.n1,cl)
        x2 = self.bn1s(self.conv1s(x)).view(b,n,self.n1,cl).sum(dim=1).view(b,1,self.n1,cl).repeat(1,n,1,1)
        x = F.relu(x1 + x2)

        x = x.view(n * b, self.n1, cl)
        cl = math.ceil(cl/2)
        x1 = self.bn2(self.conv2(x)).view(b, n, self.n1, cl)
        x2 = self.bn2s(self.conv2s(x)).view(b, n, self.n1, cl).sum(dim=1).view(b,1,self.n1,cl).repeat(1, n, 1, 1)
        x = F.relu(x1 + x2)

        x = x.view(n * b, self.n1, cl)
        cl = math.ceil(cl/2)
        x1 = self.bn3(self.conv3(x)).view(b, n, self.n2, cl)
        x2 = self.bn3s(self.conv3s(x)).view(b, n, self.n2, cl).sum(dim=1).view(b,1,self.n2,cl).repeat(1, n, 1, 1)
        x = F.relu(x1 + x2)

        x = torch.sum(torch.sum(x,dim=3),dim=1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class DeepSymmetricNetMax(nn.Module):
    def __init__(self,args):
        super(DeepSymmetricNetMax, self).__init__()
        # make sure that the number of parameters is roughly the same as in the MLPs
        self.n1 = 160
        self.n2 = 80
        # layers
        self.conv1 = nn.Conv1d(1, self.n1, kernel_size=l,stride=2, padding=l-1, padding_mode='circular')
        self.conv2 = nn.Conv1d(self.n1, self.n1, kernel_size=int(l/2),stride=2, padding=int(l/2)-1, padding_mode='circular')
        self.conv3 = nn.Conv1d(self.n1, self.n2, kernel_size=int(l/4),stride=2, padding=int(l/4)-1, padding_mode='circular')
        self.conv1s = nn.Conv1d(1, self.n1, kernel_size=l,stride=2, padding=l-1, padding_mode='circular')
        self.conv2s = nn.Conv1d(self.n1, self.n1, kernel_size=int(l/2),stride=2, padding=int(l/2)-1, padding_mode='circular')
        self.conv3s = nn.Conv1d(self.n1, self.n2, kernel_size=int(l/4),stride=2, padding=int(l/4)-1, padding_mode='circular')
        self.bn1 = torch.nn.BatchNorm1d(self.n1)
        self.bn2 = torch.nn.BatchNorm1d(self.n1)
        self.bn3 = torch.nn.BatchNorm1d(self.n2)
        self.bn1s = torch.nn.BatchNorm1d(self.n1)
        self.bn2s = torch.nn.BatchNorm1d(self.n1)
        self.bn3s = torch.nn.BatchNorm1d(self.n2)
        self.fc1 = nn.Linear(self.n2,num_labels)

        # initializations
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv1s.weight)
        torch.nn.init.xavier_uniform_(self.conv2s.weight)
        torch.nn.init.xavier_uniform_(self.conv3s.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        b = len(x)
        x = x.view(n*b,1,l)
        cl = math.ceil(l/2)
        x1 = self.bn1(self.conv1(x)).view(b,n,self.n1,cl)
        x2,_ = self.bn1s(self.conv1s(x)).view(b,n,self.n1,cl).max(dim=1)
        x2 = x2.view(b,1,self.n1,cl).repeat(1,n,1,1)
        x = F.relu(x1 + x2)

        x = x.view(n * b, self.n1, cl)
        cl = math.ceil(cl/2)
        x1 = self.bn2(self.conv2(x)).view(b, n, self.n1, cl)
        x2, _ = self.bn2s(self.conv2s(x)).view(b, n, self.n1, cl).max(dim=1)
        x2 = x2.view(b, 1, self.n1, cl).repeat(1, n, 1, 1)
        x = F.relu(x1 + x2)

        x = x.view(n * b, self.n1, cl)
        cl = math.ceil(cl/2)
        x1 = self.bn3(self.conv3(x)).view(b, n, self.n2, cl)
        x2, _ = self.bn3s(self.conv3s(x)).view(b, n, self.n2, cl).max(dim=1)
        x2 = x2.view(b, 1, self.n2, cl).repeat(1, n, 1, 1)
        x = F.relu(x1 + x2)

        x = torch.sum(torch.sum(x,dim=3),dim=1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class DeepSymmetricNetMax2(nn.Module):
    def __init__(self,args):
        super(DeepSymmetricNetMax2, self).__init__()
        # make sure that the number of parameters is roughly the same as in the MLPs
        self.n0 = 1
        self.n1 = 160
        self.n2 = 80
        # layers
        self.conv1 = nn.Conv1d(1, self.n1, kernel_size=l,stride=2, padding=l-1, padding_mode='circular')
        self.conv2 = nn.Conv1d(self.n1, self.n1, kernel_size=int(l/2),stride=2, padding=int(l/2)-1, padding_mode='circular')
        self.conv3 = nn.Conv1d(self.n1, self.n2, kernel_size=int(l/4),stride=2, padding=int(l/4)-1, padding_mode='circular')
        self.conv1s = nn.Conv1d(1, self.n1, kernel_size=l,stride=2, padding=l-1, padding_mode='circular')
        self.conv2s = nn.Conv1d(self.n1, self.n1, kernel_size=int(l/2),stride=2, padding=int(l/2)-1, padding_mode='circular')
        self.conv3s = nn.Conv1d(self.n1, self.n2, kernel_size=int(l/4),stride=2, padding=int(l/4)-1, padding_mode='circular')
        self.bn1 = torch.nn.BatchNorm1d(self.n1)
        self.bn2 = torch.nn.BatchNorm1d(self.n1)
        self.bn3 = torch.nn.BatchNorm1d(self.n2)
        self.bn1s = torch.nn.BatchNorm1d(self.n1)
        self.bn2s = torch.nn.BatchNorm1d(self.n1)
        self.bn3s = torch.nn.BatchNorm1d(self.n2)
        self.fc1 = nn.Linear(self.n2,num_labels)

        # initializations
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv1s.weight)
        torch.nn.init.xavier_uniform_(self.conv2s.weight)
        torch.nn.init.xavier_uniform_(self.conv3s.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        b = len(x)
        cl = l
        x = x.view(n*b,1,l)
        x2 = self.bn1s(self.conv1s(x.view(b,n,self.n0,cl).max(dim=1)[0]))
        cl = math.ceil(l/2)
        x2 = x2.view(b, 1, self.n1, cl).repeat(1, n, 1, 1)
        x1 = self.bn1(self.conv1(x)).view(b,n,self.n1,cl)

        x = F.relu(x1 + x2)

        x = x.view(n * b, self.n1, cl)
        x2 = self.bn2s(self.conv2s(x.view(b, n, self.n1, cl).max(dim=1)[0]))
        cl = math.ceil(cl / 2)
        x1 = self.bn2(self.conv2(x)).view(b, n, self.n1, cl)
        x2 = x2.view(b, 1, self.n1, cl).repeat(1, n, 1, 1)
        x = F.relu(x1 + x2)

        x = x.view(n * b, self.n1, cl)
        x2 = self.bn3s(self.conv3s(x.view(b, n, self.n1, cl).max(dim=1)[0]))
        cl = math.ceil(cl / 2)
        x1 = self.bn3(self.conv3(x)).view(b, n, self.n2, cl)
        x2 = x2.view(b, 1, self.n2, cl).repeat(1, n, 1, 1)
        x = F.relu(x1 + x2)

        x = torch.sum(torch.sum(x,dim=3),dim=1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

class Aittala(nn.Module):
    def __init__(self,args):
        super(Aittala, self).__init__()
        # make sure that the number of parameters is roughly the same as in the MLPs
        self.n1 = 160
        self.n2 = 80
        # layers
        self.conv1 = nn.Conv1d(1, self.n1, kernel_size=l,stride=2, padding=l-1, padding_mode='circular')
        self.conv2 = nn.Conv1d(2*self.n1, self.n1, kernel_size=int(l/2),stride=2, padding=int(l/2)-1, padding_mode='circular')
        self.conv3 = nn.Conv1d(2*self.n1, self.n2, kernel_size=int(l/4),stride=2, padding=int(l/4)-1, padding_mode='circular')
        self.fc1 = nn.Linear(2*self.n2,num_labels)
        self.bn1 = torch.nn.BatchNorm1d(self.n1)
        self.bn2 = torch.nn.BatchNorm1d(self.n1)
        self.bn3 = torch.nn.BatchNorm1d(self.n2)


        # initializations
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        b = len(x)

        x = x.view(n*b,1,l)
        cl = math.ceil(l/2)
        x1 = self.bn1(self.conv1(x)).view(b,n,self.n1,cl)
        x1 = F.relu(x1)
        x2,_ = torch.max(x1,dim=1,keepdim=True)
        x2 = x2.repeat(1,n,1,1)
        x = torch.cat((x1,x2),dim = 2)

        x = x.view(n * b, 2*self.n1, cl)
        cl = math.ceil(cl/2)
        x1 = self.bn2(self.conv2(x)).view(b, n, self.n1, cl)
        x1 = F.relu(x1)
        x2, _ = torch.max(x1, dim=1, keepdim=True)
        x2 = x2.repeat(1, n, 1, 1)
        x = torch.cat((x1, x2), dim=2)

        x = x.view(n * b, 2*self.n1, cl)
        cl = math.ceil(cl/2)
        x1 = self.bn3(self.conv3(x)).view(b, n, self.n2, cl)
        x1 = F.relu(x1)
        x2, _ = torch.max(x1, dim=1, keepdim=True)
        x2 = x2.repeat(1, n, 1, 1)
        x = torch.cat((x1, x2), dim=2)

        x = torch.sum(torch.sum(x,dim=3),dim=1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
