import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import random
from torch.utils import data
import argparse
import os, shutil
from ResNet_model import ResNet
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
import argparse

torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device('cuda')

#Data augmentation function
def Meiosis(signal, Q, rand_subs_stre, split):
    num_subs = signal.shape[0]
    signal_len = signal.shape[-2]
    new_signal = []
    new_signal1 = []
    new_signal2 = []
    for i in range(0, 22):  ###
        si = rand_subs_stre[i]
        sj = rand_subs_stre[i + Q]
        xi = np.concatenate([signal[si, :, :, :split], signal[sj, :, :, split:]], axis=2)
        xj = np.concatenate([signal[sj, :, :, :split], signal[si, :, :, split:]], axis=2)
        new_signal1.append(xi)
        new_signal2.append(xj)
    new_signal = new_signal1 + new_signal2
    new_signal.append(signal[rand_subs_stre[-1]])
    new_signal = np.array(new_signal)
    return new_signal

# Contrast loss function
def constrast_loss(x, criterion, tau):
    LARGE_NUM = 1e9
    x = F.normalize(x, dim=-1)
    num = int(x.shape[0] / 2)
    hidden1, hidden2 = torch.split(x, num)
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = torch.arange(0, num).to('cuda')
    masks = F.one_hot(torch.arange(0, num), num).to('cuda')
    logits_aa = torch.matmul(hidden1, hidden1_large.T) / tau
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(hidden2, hidden2_large.T) / tau
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, hidden2_large.T) / tau
    logits_ba = torch.matmul(hidden2, hidden1_large.T) / tau
    loss_a = criterion(torch.cat([logits_ab, logits_aa], 1),
                       labels)
    loss_b = criterion(torch.cat([logits_ba, logits_bb], 1),
                       labels)
    loss = torch.mean(loss_a + loss_b)
    return loss, labels, logits_ab

'''Because each subject in the SEED dataset has repeated the experiment on a video
 for three times, this code adds a part of algorithm in _eval and _train to ensure that multiple 
 EEG signals of the same subject do not appear in each group sample'''

#Test function
def _eval(model, test_loader, Q, tau):
    iters = 0
    total_acc = 0
    total_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            iters += 1
            bs, ss = x.shape[0], x.shape[1]
            rand_subs_stre = random.sample(range(0, ss), ss)
            change = []
            change_sub = []
            for i in range(0, int((ss - 1) / 2)):
                if rand_subs_stre[i] // 3 == rand_subs_stre[i + int((ss - 1) / 2)] // 3:
                    change.append(i)
                    change_sub.append(rand_subs_stre[i])
                i = i + 1
            if len(change) == 1:
                if rand_subs_stre[change[0]] // 3 == rand_subs_stre[-1] // 3:
                    if change[0] == ((ss - 1) / 2) - 1:
                        temp = rand_subs_stre[change[0]]
                        rand_subs_stre[change[0]] = rand_subs_stre[change[0] - 1]
                        rand_subs_stre[change[0] - 1] = temp
                    else:
                        temp = rand_subs_stre[change[0]]
                        rand_subs_stre[change[0]] = rand_subs_stre[change[0] + 1]
                        rand_subs_stre[change[0] + 1] = temp
                else:
                    temp = rand_subs_stre[change[0]]
                    rand_subs_stre[change[0]] = rand_subs_stre[-1]
                    rand_subs_stre[-1] = temp
            elif len(change) >= 1:
                t = change.pop(0)
                change.append(t)
                for c in range(0, len(change)):
                    rand_subs_stre[change[c]] = change_sub[c]
            split = random.randint(1, 200 - 2)
            x = x.cpu().numpy()
            #    y = []
            #   z = []
            for i in range(bs):
                x[i] = Meiosis(x[i], Q, rand_subs_stre, split)
            groups = []
            groups_1 = []
            groups_2 = []
            rand_subs = random.sample(range(ss - 1), 2 * Q)
            rand_subs1 = rand_subs[:Q]
            rand_subs2 = rand_subs[Q:]
            for i in range(bs):
                groups_1.append(x[i, rand_subs1])
            for i in range(bs):
                groups_2.append(x[i, rand_subs2])
            groups = groups_1 + groups_2
            groups = np.concatenate(groups)
            groups = groups.reshape(-1, groups.shape[-3], groups.shape[-2], groups.shape[-1])
            if groups.shape[0] % (2 * Q) != 0:
                groups = groups[:-2]  ###
            groups = torch.tensor(groups, dtype=torch.float, device=device)
            out = model(groups, 'contrast')
            out = torch.reshape(out, (-1, Q, out.shape[-1]))
            out = torch.max(out, dim=1)[0]
            tem_loss, lab_con, logits_ab = constrast_loss(out, criterion, tau)
            _, log_p = torch.max(logits_ab.data, 1)
            loss = tem_loss
            evaluation_batch = ((log_p == lab_con).cpu().numpy() * 1)
            acc = sum(evaluation_batch) / evaluation_batch.shape[0]
            total_acc += acc
            total_loss += loss
        test_acc = total_acc / iters
        test_loss = total_loss / iters
        # print(f'Epoch: {epoch}, Train_loss: {test_loss:.4f}, Test_acc: {test_acc:.4f}')
    return test_acc, test_loss

#Training function
def _train(model, train_loader, optimizer, epoch, Q, tau):
    model.train()
    evaluation = []
    train_losses = []
    train_acces = []
    iters = 0
    for x, y in train_loader:
        iters += 1
        bs, ss = x.shape[0], x.shape[1]
        rand_subs_stre = random.sample(range(0, ss), ss)
        change = []
        change_sub = []
        for i in range(0, int((ss - 1) / 2)):
            if rand_subs_stre[i] // 3 == rand_subs_stre[i + int((ss - 1) / 2)] // 3:
                change.append(i)
                change_sub.append(rand_subs_stre[i])
            i = i + 1
        if len(change) == 1:
            if rand_subs_stre[change[0]] // 3 == rand_subs_stre[-1] // 3:
                if change[0] == ((ss - 1) / 2) - 1:
                    temp = rand_subs_stre[change[0]]
                    rand_subs_stre[change[0]] = rand_subs_stre[change[0] - 1]
                    rand_subs_stre[change[0] - 1] = temp
                else:
                    temp = rand_subs_stre[change[0]]
                    rand_subs_stre[change[0]] = rand_subs_stre[change[0] + 1]
                    rand_subs_stre[change[0] + 1] = temp
            else:
                temp = rand_subs_stre[change[0]]
                rand_subs_stre[change[0]] = rand_subs_stre[-1]
                rand_subs_stre[-1] = temp
        elif len(change) >= 1:
            t = change.pop(0)
            change.append(t)
            for c in range(0, len(change)):
                rand_subs_stre[change[c]] = change_sub[c]
        split = random.randint(1, 200 - 2)
        x = x.cpu().numpy()
        for i in range(bs):
            x[i] = Meiosis(x[i], Q, rand_subs_stre, split)
        groups = []
        groups_1 = []
        groups_2 = []
        rand_subs = random.sample(range(ss - 1), 2 * Q)
        rand_subs.sort()
        rand_subs1 = rand_subs[Q:]
        rand_subs2 = rand_subs[:Q]
        for i in range(bs):
            groups_1.append(x[i, rand_subs1])
        for i in range(bs):
            groups_2.append(x[i, rand_subs2])
        groups = groups_1 + groups_2
        groups = np.concatenate(groups)
        groups = groups.reshape(-1, groups.shape[-3], groups.shape[-2], groups.shape[-1])
        if groups.shape[0] % (2 * Q) != 0:
            groups = groups[:-2]  ###
        groups = torch.tensor(groups, dtype=torch.float, device=device)
        out = model(groups, 'contrast')
        out = torch.reshape(out, (-1, Q, out.shape[-1]))
        out = torch.max(out, dim=1)[0]
        optimizer.zero_grad()
        tem_loss, lab_con, logits_ab = constrast_loss(out, criterion, tau)
        _, log_p = torch.max(logits_ab.data, 1)
        loss = tem_loss
        evaluation_batch = ((log_p == lab_con).cpu().numpy() * 1)
        loss.backward()
        loss = loss.item()
        optimizer.step()
        acc = sum(evaluation_batch) / evaluation_batch.shape[0]
        train_losses.append(loss)
        train_acces.append(acc)
        print(f'Epoch: {epoch}, batch: {iters}, Train_loss: {loss:.4f}, Train_acc: {acc:.4f}')
    return train_acces, train_losses

#SSL training function
def _train_epochs(model, train_loader, test_loader, epochs, Q, tau):
    optimizer = optim.Adam(model.parameters(), lr)
    print('开始预训练')
    test_acc, test_loss = _eval(model, test_loader, Q, tau)
    print(f'Epoch: {0}, Train_loss: {test_loss:.4f}, Test_acc: {test_acc:.4f}')
    test_acces = [test_acc]
    test_losses = [test_loss]
    train_acces = []
    train_losses = []
    for epoch in range(1, epochs + 1):
        train_acces_tem, train_losses_tem = _train(model, train_loader, optimizer, epoch, Q, tau)
        if epoch % 1 == 0:
            torch.save(model,
                       os.path.join(saved_models_dir, saved_models_dir + ';epoch={}.pth'.format(str(epoch).zfill(3))))
        train_acces.extend(train_acces_tem)
        train_losses.extend(train_losses_tem)
        # 每个epoch测试一次
        test_acc, test_loss = _eval(model, test_loader, Q, tau)
        test_losses.append(test_loss)
        test_acces.append(test_acc)
        print(f'Epoch: {epoch}, Train_loss: {test_loss:.4f}, Test_acc: {test_acc:.4f}')
    return test_acces, test_losses, train_acces, train_losses


import numpy as np

device = "cuda"
#Import data and set hyper-parameter
x_train_path = 'x_train_SEED.npy'
x_test_path = 'x_test_SEED.npy'
y_train_path = 'y_train_SEED.npy'
y_test_path = 'y_test_SEED.npy'

x_train = np.load(x_train_path)
x_test = np.load(x_test_path)
y_train = np.load(y_train_path)
y_test = np.load(y_test_path)

x_train = torch.tensor(x_train, dtype=torch.float)
x_test = torch.tensor(x_test, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
print(x_train.shape)

model = ResNet()
model = nn.DataParallel(model)
criterion = nn.CrossEntropyLoss().to(device)
import torch.optim as optim

epochs = 4000
tau = 0.1
P = 16
Q = 2
lr = 0.001

#Set model storage address
model_name = 'ResNet_SEED'
saved_models_dir = 'Pretrained_'+model_name+';totalepochs='+format(str(epochs).zfill(3))+';P='+format(str(P).zfill(3))+';Q='+format(str(Q).zfill(2))+';tau='+format(tau,'.2E')+';lr='+format(lr,'.2E')
ssl_logs = os.path.join(saved_models_dir,'ssl_logs')

os.makedirs(ssl_logs, exist_ok=True)
if not os.path.exists(ssl_logs):
    os.makedirs(ssl_logs)


train_dataset = data.TensorDataset(x_train, y_train)
train_loader = data.DataLoader(train_dataset, P, shuffle=True)
test_dataset = data.TensorDataset(x_test, y_test)
test_loader = data.DataLoader(test_dataset, P, shuffle=True)

test_acces, test_losses, train_acces, train_losses = _train_epochs(model,
                                                                   train_loader,
                                                                   test_loader,
                                                                   epochs,
                                                                   Q,
                                                                   tau)

import pandas as pd

name_path_per_ssl_epoch = os.path.join(ssl_logs, saved_models_dir + ';ssl_acces.xls')
per_ssl_accs = pd.DataFrame(test_acces, columns=['ssl_acc'])

import matplotlib.pyplot as plt

plt.plot(test_acces)
plt.legend(['pre-train_ssl'], loc='upper left')
plt.ylabel('test_acc')
plt.xlabel('epoch_of_pretrain')
plt.title(saved_models_dir + ';ssl_log', fontdict={'size': 9})
plt.title(saved_models_dir)
plt.savefig(os.path.join(ssl_logs, saved_models_dir + '.jpg'))
plt.show()