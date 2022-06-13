import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F

from ResNet_model import ResNet

model = ResNet()

x_train = np.load('x_train_SEED.npy')
x_test = np.load('x_test_SEED.npy')
y_train = np.load('y_train_SEED.npy')
y_test = np.load('y_test_SEED.npy')

x_train = x_train.reshape(-1, 1, x_train.shape[-2], x_train.shape[-1])
x_test = x_test.reshape(-1, 1, x_test.shape[-2], x_test.shape[-1])
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

#数据加载器
batch_size = 256

train_dataset = data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print('负例比例：',1 - sum(y_test)/y_test.shape[0])
epochs = 30
lr=0.001
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#_train_epochs
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()
train_losses = []
test_losses = []
train_acces = []
test_losses = []


#先进行一次测试集测试
model.eval()
total_loss = 0
total_acc = 0
with torch.no_grad():
    for pair in test_loader:
        x, y = pair[0], pair[1]
        x = x.cuda().float().contiguous()
        y = y.cuda().long().contiguous()
        #y = y.cuda().long().contiguous()
        optimizer.zero_grad()
        #x = np.squeeze(x, 1)
        #x = np.swapaxes(x, -1,-2)
        out = model(x, 'classifier')
        acc = (torch.sum(torch.eq(torch.argmax(out, 1), y)).float()/y.shape[0]).item()
        loss = loss_fn(out, y)
        loss = loss.item()
        total_loss += loss * y.shape[0]
        total_acc += acc * y.shape[0]
        test_loss = total_loss / len(test_loader.dataset)
        test_acc = total_acc / len(test_loader.dataset)
test_acces = [test_acc]
test_losses = [test_loss]
print(f'Epoch {0}, Test loss {test_loss:.4f}, Test acc {test_acc:.4f}')

#开始训练
train_losses = []
train_acces = []
for epoch in range(1, epochs+1):
    model.train()
    #每一个batch
    train_losses_tem = []
    train_acces_tem = []
    num=0
    for pair in train_loader:
        num+=1
        x, y = pair[0], pair[1]
        x = x.cuda().float().contiguous()
        y = y.cuda().long().contiguous()
        optimizer.zero_grad()
        out = model(x, 'classifier')
        acc = (torch.sum(torch.eq(torch.argmax(out, 1), y)).float()/y.shape[0]).item()
        loss = loss_fn(out, y)
        loss.backward()
        loss = loss.item()
        optimizer.step()
        train_losses_tem.append(loss)
        train_acces_tem.append(acc)
        print(f'Epoch: {epoch}, batch: {num}, Train_loss: {loss:.4f}, Train_acc: {acc:.4f}')
    train_acces.extend(train_acces_tem)
    train_losses.extend(train_losses_tem)

    #测试一次
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for pair in test_loader:
            x, y = pair[0], pair[1]
            x = x.cuda().float().contiguous()
            y = y.cuda().long().contiguous()
            optimizer.zero_grad()
            out = model(x, 'classifier')
            acc = (torch.sum(torch.eq(torch.argmax(out, 1), y)).float()/y.shape[0]).item()
            loss = loss_fn(out, y)
            loss = loss.item()
            total_loss += loss * y.shape[0]
            total_acc += acc * y.shape[0]
            test_loss = total_loss / len(test_loader.dataset)
            test_acc = total_acc / len(test_loader.dataset)
    print(f'Epoch {epoch}, Test loss {test_loss:.4f}, Test acc {test_acc:.4f}')
    test_losses.append(test_loss)
    test_acces.append(test_acc)

import matplotlib.pyplot as plt
plt.plot(test_acces)
plt.show()
print(sum(test_acces[-5:])/5)
#torch.save(model,#.state_dict(),
 #   ';epoch={}.pth'.format(str(epoch).zfill(3)))