import scipy.io as sio
import argparse
import os
import numpy as np
import pandas as pd
import random
import time
import pickle
import scipy.io as scio
import torch
import torch.nn.functional as F
np.random.seed(0)
window_size = 200

def windows(data, size):
    start = 0
    while ((start + size) <= data.shape[0]):
        yield int(start), int(start + size)
        start += size
#读标签
dataset_dir = 'C:\DEAP\Preprocessed_EEG'
record_list = [task for task in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, task))][:-2]
labels_root = os.path.join(dataset_dir, 'label.mat')
labels = scio.loadmat(labels_root)['label'][0] + 1

DATA = []
LABEL = []

#Get name of video
keys_array = []
for record in record_list:
    print(record)
    file = os.path.join(dataset_dir, record)
    keys = list(scio.loadmat(file).keys())[3:]
    keys_array.append(keys)
keys_array = np.array(keys_array)
#keys_array.shape
for i in range(keys_array.shape[1]):
    sample_file = os.path.join(dataset_dir, record_list[0])
    sample_mat = scio.loadmat(sample_file)
    sample_keys = list(sample_mat.keys())[3:]
#先取一个文件为基准
for i in range(len(sample_keys)): #对电影循环
    time=-1
    video_data = []
    video_label = []
    for j in range(len(record_list)): #对人循环
        time+=1
        sub = int(time/3)
        times = time%3
        file_name = record_list[j]
        sub_video_name = keys_array[j,i]
        label = labels[i]
        print('label:', label, 'sub:', sub, 'times:',times, 'file_name:',file_name, 'sub_video_name:',sub_video_name)
        file_root = os.path.join(dataset_dir, file_name)
        file_data = scio.loadmat(file_root)
        data_tem = file_data[sub_video_name]#[channel]
        label_tem = label#, sub, times, i] #(标签， 被试， 观看次数， 视频)
        video_data.append(data_tem)
        video_label.append(label_tem)
    data_in = np.array(video_data)
    label_in = np.array(video_label)
    for (start, end) in windows(data_in[0,0], window_size):
        if ((len(data_in[0,0,start:end]) == window_size)):
            segments = data_in[:,:,start:end]
            segments = F.normalize(torch.tensor(segments), dim=2)
            segments = segments.numpy()
            #label, sub, times, video, start
            labels = label_in#np.column_stack((label_in, np.array([start for k in range(label_in.shape[0])])))
            DATA.append(segments)
            LABEL.append(labels)
            print(len(DATA), len(LABEL))

DATA = np.array(DATA)
DATA = np.expand_dims(DATA,-3)
LABEL = np.array(LABEL)

#Get Training , Test and validation set
from sklearn.model_selection import train_test_split
x_train_SEED, x_test_SEED, y_train_SEED, y_test_SEED = train_test_split(DATA, LABEL, random_state=42, test_size=0.3)
x_test_SEED, x_val_SEED, y_test_SEED, y_val_SEED = train_test_split(x_test_SEED, y_test_SEED, random_state=42, test_size=0.5)
np.save('DATA_SEED.npy',DATA)
np.save('LABEL_SEED.npy',LABEL)
np.save('x_train_SEED.npy',x_train_SEED)
np.save('x_test_SEED.npy',x_test_SEED)
np.save('x_val_SEED.npy',x_val_SEED)
np.save('y_train_SEED.npy',y_train_SEED)
np.save('y_test_SEED.npy',y_test_SEED)
np.save('y_val_SEED.npy',y_val_SEED)