import random

import h5py
import healpy as hp
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
from torch.utils.data import DataLoader


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def create_dataframe(hdf_paths):
    M, dec, ra, snr = np.zeros((0, 42)), [], [], []
    for n, hdf_path in enumerate(hdf_paths):
        with h5py.File(hdf_path, 'r') as f:
            h1_strain = f['injection_samples']['h1_strain'][()]
            l1_strain = f['injection_samples']['l1_strain'][()]
            v1_strain = f['injection_samples']['v1_strain'][()]
            k1_strain = f['injection_samples']['k1_strain'][()]
            dec_i = f['injection_parameters']['dec'][()]
            ra_i = f['injection_parameters']['ra'][()]
            snr_i = f['injection_parameters']['injection_snr'][()]

        dec += list(dec_i)
        ra += list(ra_i)
        snr += list(snr_i)
        
        l = len(h1_strain)
        strains = {0:h1_strain, 1:l1_strain, 2:v1_strain, 3:k1_strain}

        # Take mean 0
        for i in range(l):
            for j in range(4):
                strains[j][i] -= np.mean(strains[j][i])

        hilbert_h1 = scipy.signal.hilbert(h1_strain)
        hilbert_l1 = scipy.signal.hilbert(l1_strain)
        hilbert_v1 = scipy.signal.hilbert(v1_strain)
        hilbert_k1 = scipy.signal.hilbert(k1_strain)
        hilberts = {0:hilbert_h1, 1:hilbert_l1, 2:hilbert_v1, 3:hilbert_k1}

        # Initializing input features matrix
        features = np.zeros((l,42))

        for i in range(l):
            features_i = []

            for d1 in range(4):
                # for (v)(vi)
                coal_time1 = np.argmax(np.abs(strains[d1][i]))
                amp1 = strains[d1][i][coal_time1]
                phase1 = hilberts[d1][i][coal_time1].imag

                for d2 in range(d1+1,4):
                    # (i) Arrival time delays of signals
                    # (ii) Maximum cros-correlation values of signals
                    corr = np.correlate(strains[d1][i], strains[d2][i], 'full')
                    corr_argmax = np.argmax(abs(corr))

                    features_i.append(corr_argmax-len(strains[d1][i])+1)
                    features_i.append(corr[corr_argmax])

                    # (iii) Arrival time delays of analytic signal
                    # (iv) Maximum cross-correlation values of analytic signals
                    corr_h = np.correlate(hilberts[d1][i],hilberts[d2][i],'full')
                    corr_h = np.abs(corr_h)
                    corr_h_argmax = np.argmax(corr_h)

                    features_i.append(corr_h.argmax()-len(strains[d1][i])+1)
                    features_i.append(corr_h[corr_h_argmax])

                    # (v) Ratios of average instantaneous amplitudes around merger
                    # (vi) Phase lags around merger
                    coal_time2 = np.argmax(np.abs(strains[d2][i]))
                    amp2 = strains[d2][i][coal_time2]
                    phase2 = hilberts[d2][i][coal_time2].imag

                    features_i.append(amp1/amp2)
                    features_i.append(phase1-phase2)

                    # (vii) Complex Correlation Coefficients
                    ccc = scipy.stats.pearsonr(strains[d1][i], strains[d2][i])[0]
                    features_i.append(ccc)
                    
            features[i] = features_i
	
    M = np.concatenate([M, features])

    dec, ra, snr = np.array(dec), np.array(ra), np.array(snr)

    # Standarization
    if M.shape[0] > 1:
        for i in range(M.shape[1]):
            M[:,i] -= np.mean(M[:,i])
            M[:,i] /= np.std(M[:,i])

    columns = ["Delay HL","Max-Corr HL","Delay-Ana HL","Max-Corr-Ana HL",
               "Amp-Ratio HL","Phase-lag HL","Corr-Coef HL",
               "Delay HV","Max-Corr HV","Delay-Ana HV","Max-Corr-Ana HV",
               "Amp-Ratio HV","Phase-lag HV","Corr-Coef HV",
               "Delay HK","Max-Corr HK","Delay-Ana HK","Max-Corr-Ana HK",
               "Amp-Ratio HK","Phase-lag HK","Corr-Coef HK",
               "Delay LV","Max-Corr LV","Delay-Ana LV","Max-Corr-Ana LV",
               "Amp-Ratio LV","Phase-lag LV","Corr-Coef LV",
               "Delay LK","Max-Corr LK","Delay-Ana LK","Max-Corr-Ana LK",
               "Amp-Ratio LK","Phase-lag LK","Corr-Coef LK",
               "Delay VK","Max-Corr VK","Delay-Ana VK","Max-Corr-Ana VK",
               "Amp-Ratio VK","Phase-lag VK","Corr-Coef VK"
              ]
    df = pd.DataFrame(M, columns=columns)
    df['dec'] = dec
    df['ra'] = ra
    df['snr'] = snr

    return df

def drop_KAGRA(df):
    df = df.drop(["Delay HK","Delay LK","Delay VK",
                  "Max-Corr HK","Max-Corr LK","Max-Corr VK",
                  "Delay-Ana HK","Delay-Ana LK","Delay-Ana VK",
                  "Max-Corr-Ana HK","Max-Corr-Ana LK","Max-Corr-Ana VK",
                  "Amp-Ratio HK","Amp-Ratio LK","Amp-Ratio VK",
                  "Phase-lag HK","Phase-lag LK","Phase-lag VK",
                  "Corr-Coef HK","Corr-Coef LK","Corr-Coef VK"
                 ], axis=1)
    return df

def create_train_dataloader(model, df, arr, labels, n_sectors, training_size, batch_size):
    if (model=='MLP' and training_size >= len(df)) \
                                    or (model=='TCN' and training_size >= len(arr)):
        raise RuntimeError('Training size must be smaller than the number of samples.')

    if model=='MLP':
        df_ = df.drop(['dec', 'ra', 'snr'], axis=1)
        X_train, X_val = df_[:training_size].values, df_[training_size:len(labels)].values
    else:
        X_train, X_val = arr[:training_size], arr[training_size:len(labels)]

    y_train, y_val = labels[:training_size], labels[training_size:len(labels)]

    X_train, X_val = torch.from_numpy(X_train).float(), torch.from_numpy(X_val).float()
    y_train, y_val = torch.from_numpy(y_train).long(), torch.from_numpy(y_val).long()

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=2
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size, shuffle=False, num_workers=2
    )
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    return dataloaders_dict

def create_labels_from_hdf(hdf_paths, n_sectors, HEALPix=False):
    dec, ra, snr = [], [], []
    for i, hdf_path in enumerate(hdf_paths):
        with h5py.File(hdf_path, 'r') as f:
            dec_i = f['injection_parameters']['dec'][()]
            ra_i = f['injection_parameters']['ra'][()]
            snr_i = f['injection_parameters']['injection_snr'][()]

        dec += list(dec_i)
        ra += list(ra_i)
        snr += list(snr_i)
    dec, ra, snr = np.array(dec), np.array(ra), np.array(snr)

    if not HEALPix:
        delta = np.pi/int((n_sectors/2)**(0.5))
        dec_label = ((dec+np.pi/2)/delta).astype(int)
        ra_label = ((ra/delta)).astype(int)
        labels = 2*int((n_sectors/2)**(0.5))*dec_label+ra_label
    else:
        Nside = int((n_sectors/12)**0.5)
        labels = []
        for i in range(len(dec)):
            labels.append(hp.pixelfunc.ang2pix(Nside, dec[i]+np.pi/2, ra[i], nest=True))
        labels = np.array(labels)
    return labels

def create_array_from_hdf(input_paths, use_KAGRA):
    h1_strain, l1_strain = np.zeros((0,512)), np.zeros((0,512))
    v1_strain, k1_strain = np.zeros((0,512)), np.zeros((0,512))

    for i, input_path in enumerate(input_paths):
        with h5py.File(input_path, 'r') as f:
            h1_strain_i = f['injection_samples']['h1_strain'][()]
            l1_strain_i = f['injection_samples']['l1_strain'][()]
            v1_strain_i = f['injection_samples']['v1_strain'][()]
            k1_strain_i = f['injection_samples']['k1_strain'][()]

            h1_strain = np.concatenate([h1_strain, h1_strain_i])
            l1_strain = np.concatenate([l1_strain, l1_strain_i])
            v1_strain = np.concatenate([v1_strain, v1_strain_i])
            k1_strain = np.concatenate([k1_strain, k1_strain_i])

    l = len(h1_strain)
    strains = {0:h1_strain, 1:l1_strain, 2:v1_strain, 3:k1_strain}

    # Take mean 0 and normalize
    for i in range(l):
        for j in range(4):
            strains[j][i] -= np.mean(strains[j][i])
            strains[j][i] /= (np.abs(strains[j][i])).max()

    if use_KAGRA:
        return np.dstack([h1_strain, l1_strain, v1_strain, k1_strain])
    else:
        return np.dstack([h1_strain, l1_strain, v1_strain])

def create_test_dataloader(arr, labels, n_sectors, batch_size):
    X_test = torch.from_numpy(arr).float()
    y_test = torch.from_numpy(labels).long()
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return test_dataloader
