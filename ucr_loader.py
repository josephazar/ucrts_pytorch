#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:59:27 2023

@author: josephazar
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time


def readucr(filename):
    data = []
    labels = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            if "," in line:
                fields = line.strip().split(',')
            else:
                fields = line.strip().split('')
            labels.append(int(fields[0]))
            data.append([float(x) for x in fields[1:]])
    return np.array(data),np.array(labels)


def pre_processing_dataset(file_to_read = "Adiac"):
    train_path = file_to_read + '/' + file_to_read + '_TRAIN.txt'
    test_path = file_to_read + '/' + file_to_read + '_TEST.txt'
    
    x_train, y_train = readucr(train_path)
    x_test, y_test = readucr(test_path)
    y_train = y_train - 1
    y_test = y_test - 1

    
    np.save(file_to_read + '/'+'train_x.npy', x_train)
    np.save(file_to_read + '/'+'train_y.npy', y_train)
    np.save(file_to_read + '/'+'test_x.npy', x_test)
    np.save(file_to_read + '/'+'test_y.npy', y_test)

    
    
class UCRLoader(Dataset):
    def __init__(self, file_path):
        start_time = time.time()
        self.buffer_x = np.load(file_path + '_x.npy')
        self.buffer_y = np.load(file_path + '_y.npy')
        #print('X shape:', self.buffer_x.shape)
        #print('Y shape:', self.buffer_y.shape)
        end_time = time.time()
        elapsed_time = end_time - start_time
        #print('Data loading is done. Takes {:3.2f}s.'.format(elapsed_time))
        
    def __len__(self):
        return len(self.buffer_x)
    
    def __getitem__(self, idx):
        return {'data': self.buffer_x[idx], 'label': self.buffer_y[idx]}

file_to_read = "Adiac"
def main():
    start_time = time.time()
    pre_processing_dataset(file_to_read)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Data processing is done. Takes {:3.2f}s.'.format(elapsed_time))

def sample_label_size():
    dataset = UCRLoader(file_to_read + '/' + 'train')
    input_shape = dataset.buffer_x.shape
    num_classes = len(np.unique(dataset.buffer_y))
    return input_shape[1], num_classes
    
def train_dataset():
    return UCRLoader(file_to_read + '/' + 'train')


def test_dataset():
    return UCRLoader(file_to_read + '/' + 'test')


if __name__ == '__main__':
    main()
    




