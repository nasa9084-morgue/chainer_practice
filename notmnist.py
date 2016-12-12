import sys, os, pickle

import numpy as np
import numpy.random as rd
from scipy.misc import imread

import matplotlib.pyplot as plt

image_size = 28
depth = 255


def unpickle(fname):
    with open(fname, 'rb') as f:
        d = pickle.load(f)
    return d


def topickle(fname, obj):
    with open(fname, 'wb') as f:
        pickle.Pickler(f, protocol=2).dump(obj)


def count_empty_file(directory):
    cnt = 0
    for f in os.listdir(directory):
        if os.stat(os.path.join(directory, f)).st_size == 0:
            cnt += 1
    return cnt


alpha2num = {a: i for a, i in zip('ABCDEFGHIJ', range(10))}
num2alpha = {i: a for i, a in alpha2num.items()}


for root in ['notMNIST_large']:
    dirs = [os.path.join(root, d)
            for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d))]
    file_count = 0
    for directory in dirs:
        label_name = os.path.basename(directory)
        file_list = os.listdir(directory)
        file_counter += len(file_list) - count_empty_file(directory)

    dataset = np.ndarray(
        shape=(file_counter, image_size*image_size),
        dtype=np.float32
    )
    labels = np.ndarray(shape=(file_counter), dtype=np.int)

    last_num = 0

    for directory in dirs:
        file_list = os.listdir(directory)
        file_count = len(file_list) - count_empty_file(directory)

        label_name = os.path.basename(directory)
        labels[last_num:(last_num + file_count)] = alpha2num[label_name]

        skip = 0
        for i, f in enumerate(file_list):
            if os.stat(os.path.join(directory, f)).st_size == 0:
                skip += 1
                continue
            try:
                data = imread(os.path.join(directory, f))
                data = data.astype(np.float32)
                data /= depth
                dataset[last_num + i - skip, :] = data.flatten()
            except:
                skip += 1
                print('error{}'.format(f))
                continue
        last_num += i - skip

    notmnist = {}
    notmnist['data'] = dataset
    notmnist['target'] = labels
    to_pickle('{}.pkl'.format(root), notmnist)
