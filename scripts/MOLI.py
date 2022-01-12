"""
The model architecture of MOLI and related utility functions.
This file is used internally by benchmark_MOLI_drug.py
"""

import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from torch.utils.data import Dataset
import numpy as np
import time
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class MOLI(nn.Module):
    def __init__(self, df_dict, h_dim, drop_rate, dim_out):
        super(MOLI, self).__init__()
        self.encoders = nn.ModuleDict()
        self.df_dict = df_dict
        for key in df_dict.keys():
            self.encoders[key] = torch.nn.Sequential(
                nn.Linear(df_dict[key], h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(drop_rate))
        self.fc = torch.nn.Sequential(
            nn.Linear(h_dim * len(df_dict), dim_out),
            nn.Dropout(drop_rate))

    def forward(self, x):
        output = []
        for key in self.df_dict.keys():
            output.append(self.encoders[key](x[key]))
        output = torch.cat(output, dim=1)
        output = self.fc(output)
        return output


class MultiOmicDataset(Dataset):
    def __init__(self, df_dict, target_data):
        self.df_dict = df_dict
        self.target_data = target_data

    def __getitem__(self, index):
        """ Returns: tuple (sample, target) """
        data = {}
        for key in self.df_dict.keys():
            data[key] = torch.from_numpy(self.df_dict[key].iloc[index, :].values).float().to(device)
        target = self.target_data.reshape(-1, 1)[index, :]

        return data, target

    def __len__(self):
        return len(self.target_data)


class AverageMeter:
    ''' Computes and stores the average and current value '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch, task='regression'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_mae = AverageMeter()
    avg_rmse = AverageMeter()
    avg_accuracy = AverageMeter()

    model.train()
    num_steps = len(train_loader)

    end = time.time()
    lr_str = ''

    assert task in ['regression', 'classification']

    for i, (input_, targets) in enumerate(train_loader):
        if i >= num_steps:
            break
        output = model(input_)
        loss = criterion(output, targets.float().to(device))

        targets = targets.cpu().numpy()
        confs = output.detach().cpu().numpy()
        if task == 'regression':
            avg_mae.update(mean_absolute_error(targets, confs))
            avg_rmse.update(mean_squared_error(targets, confs, squared=False))
        else:
            predicts = np.argmax(confs, 1)
            avg_accuracy.update(accuracy_score(targets, predicts))

        losses.update(loss.data.item(), targets.shape[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
    if task == 'regression':
        # print(f'{epoch} \t'
        #       f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #       f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
        #       f'MAE {avg_mae.val:.4f} ({avg_mae.avg:.4f})\t'
        #       f'RMSE {avg_rmse.val:.4f} ({avg_rmse.avg:.4f})\t' + lr_str)
        # sys.stdout.flush()

        return avg_mae.avg
    else:
        # print(f'{epoch} \t'
        #       f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #       f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
        #       f'Accuracy {avg_accuracy.val:.4f} ({avg_accuracy.avg:.4f})\t' + lr_str)
        # sys.stdout.flush()

        return avg_accuracy.avg


def inference(data_loader, model, task='regression'):
    ''' Returns predictions and targets, if any. '''
    model.eval()
    activation = nn.Softmax(dim=1)

    all_confs, all_targets = [], []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            input_, target = data

            output = model(input_)
            target = target.float().to(device)
            all_confs.append(output)

            if target is not None:
                all_targets.append(target)

    confs = torch.cat(all_confs)
    targets = torch.cat(all_targets) if len(all_targets) else None
    targets = targets.cpu().numpy()
    if task == 'classification':
        confs = activation(confs)
    confs = confs.cpu().numpy()

    return confs, targets


def validate(val_loader, model, task='regression'):
    confs, targets = inference(val_loader, model)

    if task == 'classification':
        predicts = np.argmax(confs, 1)
        accuracy = accuracy_score(targets, predicts)
        return accuracy
    else:
        mae = mean_absolute_error(targets, confs)
        rmse = mean_squared_error(targets, confs, squared=False)
        return mae


def train_loop(epochs, train_loader, val_loader, model, criterion, optimizer, task='regression'):
    train_res = []
    val_res = []
    for epoch in range(1, epochs + 1):
        train_score = train(train_loader,
                            model,
                            criterion,
                            optimizer,
                            epoch,
                            task)

        train_res.append(train_score)

        if val_loader:
            val_score = validate(val_loader, model, task=task)
            # if val_score:
            #     print(f"Epoch {epoch} validation score:{val_score:.4f}")
            # else:
            #     print(f"Epoch {epoch} validation Inf")
            sys.stdout.flush()
            val_res.append(val_score)

    return np.asarray(train_res), np.asarray(val_res)
