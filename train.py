import argparse
import os
import random

import h5py
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchvision
from torch.nn import BatchNorm1d
from torch.utils.data import DataLoader

from preprocess import (create_array_from_hdf, create_dataframe,
                        create_labels_from_hdf, create_train_dataloader,
                        drop_KAGRA, fix_seed)


def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--hdf_file", nargs='*', required=True)
    parser.add_argument("--csv_file")
    parser.add_argument("--save_csv_path")
    parser.add_argument("--HEALPix", action='store_true', help='default: False')
    parser.add_argument("--method", type=int, default=1, help="MLP: 1, TCN:2")
    parser.add_argument("--use_KAGRA", action='store_true', help='default: False')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--training_size", type=int, default=200000)
    parser.add_argument("--weights_directory", default='weights')

    args = parser.parse_args()
    return args
    
def train_model(net, dataloaders_dict, criterion, \
                optimizer, scheduler, num_epochs, n_sectors, weights_path):
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    max_acc = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs), end='\t')
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss, epoch_corrects = 0.0, 0.0
            for inputs, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss/len(dataloaders_dict[phase].dataset)
            epoch_acc = \
            (epoch_corrects.double()/len(dataloaders_dict[phase].dataset)).item()

            if phase=='train':
                train_loss.append(round(epoch_loss,5))
                train_acc.append(round(epoch_acc,5))
                print('train loss: {:.4f} acc: {:.4f}'.format(epoch_loss, epoch_acc),
                      end='\t')
            else:
                val_loss.append(round(epoch_loss,5))
                val_acc.append(round(epoch_acc,5))
                print('val loss: {:.4f} acc: {:.4f}'.format(epoch_loss, epoch_acc))

                if epoch_acc > max_acc:
                    max_acc = epoch_acc
                    torch.save(net.state_dict(), weights_path)
                    print('Saved the weights.')

        scheduler.step(epoch_loss)
    history = {'train_loss': train_loss, 'train_acc': train_acc,
               'val_loss': val_loss, 'val_acc': val_acc}
    return history

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, \
                 stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride, int(padding/2), dilation
        )
        self.bn1 = nn.BatchNorm1d(64, eps=1e-03, momentum=0.01)
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride, int(padding/2), dilation
        )
        self.bn2 = nn.BatchNorm1d(64, eps=1e-03, momentum=0.01)
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.bn1, self.prelu1, self.dropout1,
            self.conv2, self.bn2, self.prelu2, self.dropout2
        )

        if n_inputs != n_outputs:
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)
        else:
            self.downsample = None
        self.prelu = nn.PReLU()
        self.init_weights()

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample is not None:
            torch.nn.init.kaiming_normal_(self.downsample.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.prelu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += \
            [TemporalBlock(in_channels, out_channels, kernel_size, \
                           stride=1, dilation=dilation_size, \
                           padding=(kernel_size-1)*dilation_size, dropout=dropout)
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, \
                 num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = self.tcn(x)
        x = self.linear(x[:,:,-1])
        return x

class MLP(nn.Module):
    def __init__(self, n_sectors, use_KAGRA):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(42, 512) if use_KAGRA else nn.Linear(21, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, n_sectors)
        self.dropout1 = nn.Dropout(p=0.2)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()

        self.net = nn.Sequential(
            self.fc1, self.prelu1, self.dropout1,
            self.fc2, self.prelu2, self.dropout1,
            self.fc3, self.prelu3, self.dropout1,
            self.fc4
        )

    def forward(self, x):
        x = self.net(x)
        return x

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using', device)

    args = get_arguments()

    fix_seed(args.seed)
    input_size=4 if args.use_KAGRA else 3
    detector = 'HLVK' if args.use_KAGRA else 'HLV'
    model = 'MLP' if args.method==1 else 'TCN'

    print('Start preprocessing')

    if args.method==1:
        X = np.array([])
        if args.csv_file:
            df = pd.read_csv(args.csv_file)
        else:
            create_dataframe(args.hdf_file)

        if args.save_csv_path:
            os.makedirs(os.path.dirname(args.save_csv_path), exist_ok=True)
            df.to_csv(args.save_csv_path, index=None)
        if not args.use_KAGRA: df = drop_KAGRA(df)

    elif args.method==2:
        X = create_array_from_hdf(args.hdf_file, args.use_KAGRA).transpose(0,2,1)
        df = pd.DataFrame([])
    else:
        raise RuntimeError('Choose method 1 or 2.')

    if not args.HEALPix:
        n_sectors_list = [2*i*i for i in range(3,11)]
    else:
        n_sectors_list = [12*(4**i) for i in range(3)]

    print('Start training')
    for n_sectors in n_sectors_list:
        print('#'*30)
        print('Number of sectors:', n_sectors)
        print('#'*30)
        os.makedirs(args.weights_directory, exist_ok=True)
        weights_path = f'{args.weights_directory}/{model}_{detector}{n_sectors}.pth'
        labels = create_labels_from_hdf(args.hdf_file, n_sectors, args.HEALPix)
        if args.method==1:
            net = MLP(n_sectors, args.use_KAGRA)
        else:
            net = TCN(
                input_size=input_size, output_size=n_sectors, \
                num_channels=[64]*7, kernel_size=3, dropout=args.dropout
            )

        dataloaders_dict = create_train_dataloader(
            model=model, df=df, arr=X, labels=labels, n_sectors=n_sectors, \
            batch_size=args.batch_size, training_size = args.training_size
        )
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        history = train_model(net, dataloaders_dict, criterion, optimizer, \
                              scheduler, args.epoch, n_sectors, weights_path
                             )
