import argparse
import copy
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
from scipy.stats import rankdata
from torch.nn import BatchNorm1d
from torch.utils.data import DataLoader

from preprocess import (create_array_from_hdf, create_dataframe,
                        create_labels_from_hdf, create_test_dataloader,
                        drop_KAGRA, fix_seed)
from train import MLP, TCN, TemporalBlock, TemporalConvNet


def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--hdf_file", nargs='*', required=True)
    parser.add_argument("--csv_file")
    parser.add_argument("--save_csv_path")
    parser.add_argument("--HEALPix", action='store_true', help='default: False')
    parser.add_argument("--use_KAGRA", action='store_true', help='default: False')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=int, default=1, help="MLP:1, TCN:2, Combined:3")
    parser.add_argument("--weights_directory", default='weights')
    args = parser.parse_args()

    return args

def predict_TCN(test_dataloader, labels, n_sectors, \
                input_size, use_KAGRA, weights_path, device, method):
    net = TCN(
        input_size=input_size, output_size=n_sectors,
        num_channels=[64]*7, kernel_size=3, dropout=0
    )

    net.to(device)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.eval()
    y_pred = np.zeros((0,n_sectors))
    corrects = 0
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)

        if method != 3:
            corrects += torch.sum(preds == labels.data)

        y_pred = np.concatenate([y_pred, np.array(outputs.to('cpu').detach())], axis=0)

    if method==3:
        return y_pred
    else:
        accuracy = round(((corrects.double() / len(test_dataloader.dataset))).item(), 5)
        return y_pred, accuracy

def predict_MLP(X_test, labels, n_sectors, use_KAGRA, weights_path, method):
    net = MLP(n_sectors, use_KAGRA)
    net.load_state_dict(torch.load(weights_path, map_location='cpu'))
    net.eval()

    y_pred = net(torch.from_numpy(X_test).float())
    _, y_pred_max = torch.max(y_pred, 1)
    y_pred_max = np.array(y_pred_max)

    if method == 3:
        return y_pred.detach().numpy()
    else:
        accuracy = sum(labels == y_pred_max) / len(X_test)

        return y_pred.detach().numpy(), accuracy

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = get_arguments()
    fix_seed(args.seed)
    input_size=4 if args.use_KAGRA else 3
    detector = 'HLVK' if args.use_KAGRA else'HLV'

    model = 'MLP' if args.method==1 else 'TCN'

    if args.method==1 or args.method==3:
        if args.csv_file:
            df = pd.read_csv(args.csv_file)
        else:
            create_dataframe(args.hdf_file)

        if args.save_csv_path:
            os.makedirs(os.path.dirname(args.save_csv_path), exist_ok=True)
            df.to_csv(args.save_csv_path, index=None)
        if not args.use_KAGRA:
            df = drop_KAGRA(df)
        df_ = df.drop(['dec', 'ra', 'snr'], axis=1).values
    if args.method==2 or args.method==3:
        X = create_array_from_hdf(args.hdf_file, args.use_KAGRA).transpose(0,2,1)

    accs = []
    print('Start prediction')

    if not args.HEALPix:
        n_sectors_list = [2*i*i for i in range(3,11)]
    else:
        n_sectors_list = [12*(4**i) for i in range(3)]

    for n_sectors in n_sectors_list:
        weights_path = f'{args.weights_directory}/{model}_{detector}{n_sectors}.pth'
        labels = create_labels_from_hdf(args.hdf_file, n_sectors, args.HEALPix)

        if args.method==1:
            y_pred, accuracy = predict_MLP(
                df_, labels, n_sectors, args.use_KAGRA, weights_path, args.method
            )
        elif args.method==2:
            test_dataloader = create_test_dataloader(X, labels, n_sectors, 200)
            y_pred, accuracy = predict_TCN(
                test_dataloader, labels, n_sectors, input_size, \
                args.use_KAGRA, weights_path, device, args.method
            )
        elif args.method==3:
            weights_path = f'{args.weights_directory}/MLP_{detector}{n_sectors}.pth'
            y_pred1 = predict_MLP(df_, labels, n_sectors, args.use_KAGRA, \
                                  weights_path, args.method
                                 )

            weights_path = f'{args.weights_directory}/TCN_{detector}{n_sectors}.pth'
            test_dataloader = create_test_dataloader(X, labels, n_sectors, 200)
            y_pred2 = predict_TCN(
                test_dataloader, labels, n_sectors, input_size, args.use_KAGRA, \
                weights_path, device, args.method
            )

            r = 0.6
            y_pred = r*y_pred1 + (1-r)*y_pred2
            y_pred_max = np.argmax(y_pred, axis=1)

            accuracy = sum(labels == y_pred_max) / len(labels)

        else:
            raise RuntimeError('Choose method 1, 2, or 3.')

        accs.append(accuracy)

        print('n_sectors:', n_sectors, 'accuracy:', accuracy)

        if len(labels)==1 and args.method==3:
            print('true sector =', labels[0], \
                  'predicted sector =', y_pred_max, y_pred[0][y_pred_max])
    print(accs)
