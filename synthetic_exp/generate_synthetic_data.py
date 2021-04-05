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
import numpy as np
import os
from scipy import signal

def generate_data(seed=1,save=1,save_path='../data/'):
    # params
    np.random.seed(seed)
    dim = 100
    train_size = 3000
    val_size = 300
    test_size = 300
    set_size = 25
    total_size = train_size+test_size+val_size
    data = np.zeros((total_size,set_size,dim))
    labels = np.zeros(total_size)

    for i in range(total_size):
        labels[i] = np.random.randint(0,3)
        # prepare wave
        t = np.linspace(0, 1, dim)
        freq = 1 + 9*np.random.random()
        shift = np.random.random()
        height = 1 + 9*np.random.random()
        height_shift = -5 + 10* np.random.random()
        noise = np.random.normal(0,3,size = (set_size,dim))

        if labels[i] == 0:
            sig = np.sin(2 * np.pi * freq * (t + shift))
        elif labels[i] == 1:
            sig = signal.square(2 * np.pi * freq * (t + shift))
        else:
            sig = signal.sawtooth(2 * np.pi * freq * (t + shift))

        sig = height_shift + height * (sig + noise)
        data[i] = sig

    # split
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    train_labels = labels[:train_size]
    val_labels = labels[train_size:train_size + val_size]
    test_labels = labels[train_size + val_size:]
    if save:
        make_dir(save_path)
        fname = os.path.join(save_path,'synthetic_data')
        np.savez(fname,train_data = train_data,train_labels = train_labels,val_data = val_data,\
                 val_labels = val_labels,test_data = test_data,test_labels = test_labels)


def make_dir(dir_name):
    if not os.path.exists(dir_name): os.makedirs(dir_name)

generate_data()