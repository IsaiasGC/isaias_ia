import gzip
import os
from os.path import isfile, join
import numpy as np

def list_files(mnist_path):
    return [join(mnist_path,f) for f in os.listdir(mnist_path) if isfile(join(mnist_path, f))]

def read_images(data: gzip.GzipFile):
  _ = int.from_bytes(data.read(4), 'big')
  num_images = int.from_bytes(data.read(4), 'big')
  rows = int.from_bytes(data.read(4), 'big')
  cols = int.from_bytes(data.read(4), 'big')
  images = data.read()
  data_np = np.frombuffer(images, dtype=np.uint8)
  data_np = data_np.reshape((num_images, rows, cols))
  return data_np

def get_nmist_dataset(mnist_path):
    for f in list_files(mnist_path):
        if 'train-images' in f:
            with gzip.open(f, 'rb') as data:
                x_train = read_images(data)
        elif 'train-labels' in f:
            with gzip.open(f, 'rb') as data:
                train_labels = data.read()[8:]
                y_train = np.frombuffer(train_labels, dtype=np.uint8)
        if 't10k-images' in f:
            with gzip.open(f, 'rb') as data:
                x_test = read_images(data)
        elif 't10k-labels' in f:
            with gzip.open(f, 'rb') as data:
                test_labels = data.read()[8:]
                y_test = np.frombuffer(test_labels, dtype=np.uint8)
    
    return x_train, y_train, x_test, y_test        