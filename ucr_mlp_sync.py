#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 01:02:58 2023

@author: josephazar
"""


import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import ucr_loader as ucr_dataset
# Do not forget to define which dataset you are using in ucr_loader.py
ucr_dataset.pre_processing_dataset()
import argparse

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


class DNNUCRBatchNorm3Layer(nn.Module):
    def __init__(self, sample_size=150, model_size=250, label_num=35):
        super(DNNUCRBatchNorm3Layer, self).__init__()
        device = get_device()

        self.fc1 = nn.Linear(sample_size, model_size, False).to(device)
        self.bn1 = nn.BatchNorm1d(model_size, momentum=1.0, affine=True, track_running_stats=False).to(device)
        self.fc2 = nn.Linear(model_size, model_size, False).to(device)
        self.bn2 = nn.BatchNorm1d(model_size, momentum=1.0, affine=True, track_running_stats=False).to(device)
        self.fc3 = nn.Linear(model_size, label_num, False).to(device)
        self.bn3 = nn.BatchNorm1d(label_num, momentum=1.0, affine=False, track_running_stats=False).to(device)


    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.fc3(x)
        x = self.bn3(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x






def train(args, model, optimizer, train_loader, epoch, train_time_log):
    start_time = time.time()
    device = get_device()
    model.train()
    for i, batch in enumerate(train_loader):
        data, target = batch['data'].float().to(device), batch['label'].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()  # This will just update the local data which reduces communication overhead.
        i += 1
        train_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        train_correct = train_pred.eq(target.view_as(train_pred)).sum().item()

    end_time = time.time()
    elapsed_time = end_time - start_time
    train_time_log[epoch-1] = elapsed_time


def test(args, model, test_loader, epoch, test_loss_log, test_acc_log):
    device = get_device()
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            data, target = batch['data'].float().to(device), batch['label'].to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            test_correct += test_pred.eq(target.view_as(test_pred)).sum().item()
            test_total += target.shape[0]
        test_acc = float(test_correct) / float(test_total)
        test_loss /= float(test_total)
    print("Epoch {} Test Loss: {:.6f}; Test Accuracy: {:.2f}.\n".format(epoch, test_loss, test_acc))
    test_loss_log[epoch - 1] = test_loss
    test_acc_log[epoch - 1] = test_acc




def model_process(args):
    device = get_device()
    
    train_set = ucr_dataset.train_dataset()
    test_set = ucr_dataset.test_dataset()
    num_samples, num_classes = ucr_dataset.sample_label_size()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             drop_last=False)
    
    model_name = 'DNN_ucr_layer_BN_' + str(args.epochs) + '_' + str(args.model_size)  
    
    model = DNNUCRBatchNorm3Layer(sample_size=num_samples,model_size=args.model_size,label_num=num_classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    epochs = args.epochs 
    train_time_log = np.zeros(epochs)
    test_loss_log = np.zeros(epochs)
    test_acc_log = np.zeros(epochs)
    for epoch in range(1, epochs + 1):
        train(args, model, optimizer, train_loader, epoch, train_time_log)
        test(args, model, test_loader, epoch, test_loss_log, test_acc_log)
    np.savetxt('./log/' + model_name + '_train_time.log', train_time_log, fmt='%1.4f', newline=' ')
    np.savetxt('./log/' + model_name + '_test_loss.log', test_loss_log, fmt='%1.4f', newline=' ')
    np.savetxt('./log/' + model_name + '_test_acc.log', test_acc_log, fmt='%1.4f', newline=' ')


def main():
    parser = argparse.ArgumentParser(description='PyTorch 3-layer DNN on UCR dataset')
    parser.add_argument('--model-size', type=int, default=150, metavar='N',
                        help='model size for intermediate layers (default: 150)')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                        help='number of epochs to train (default: 25)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001 for BN)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    
    torch.manual_seed(args.seed)
    model_process(args)



if __name__ == '__main__':
    main()
