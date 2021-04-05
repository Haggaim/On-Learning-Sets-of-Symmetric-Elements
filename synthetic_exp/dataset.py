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
import numpy as np
from torch.utils.data import Dataset

class BumpsDataset(Dataset):
    def __init__(self,type,num_training=-1):
        data_path = '../data/synthetic_data.npz'
        alldata = np.load(data_path)
        if type == 'train':
            self.data = alldata['train_data']
            self.labels = alldata['train_labels']
            if num_training==-1:
                num_training = len(self.data)
            self.data = self.data[:num_training]
            self.labels = self.labels[:num_training]
        elif type == 'val':
            self.data = alldata['val_data']
            self.labels = alldata['val_labels']
        elif type == 'test':
            self.data = alldata['test_data']
            self.labels = alldata['test_labels']
        print('Read ' + type + ' data. size is ')
        print(self.data.shape)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx],self.labels[idx]

def get_data_loaders(args, kwargs):

    train_dataset = BumpsDataset(type='train',num_training = args.num_training)
    val_dataset = BumpsDataset(type='val',num_training = args.num_training)
    test_dataset = BumpsDataset(type='test',num_training = args.num_training)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader
