from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from PIL import Image
from pathlib import Path
# from keras.utils.data_utils import get_file
# from keras import backend as K
import pandas as pd
import h5py



googlefolder = "../googlefolder/"
#
# def get_unzip_file(fname,
#                    origin,
#                    untar=False,
#                    md5_hash=None,
#                    file_hash=None,
#                    cache_subdir='datasets',
#                    hash_algorithm='auto',
#                    extract=False,
#                    archive_format='auto',
#                    cache_dir=None):
#     import gzip
#     import shutil
#     get_file()
#     with open('file.txt', 'rb') as f_in, gzip.open('file.txt.gz', 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out)


def load_data():
    """Loads PCam dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """


    y_train = h5py.File(googlefolder+'camelyonpatch_level_2_split_train_y.h5', 'r')["y"]
    x_valid = h5py.File(googlefolder+'camelyonpatch_level_2_split_valid_x.h5', 'r')["x"]
    y_valid = h5py.File(googlefolder+'camelyonpatch_level_2_split_valid_y.h5', 'r')["y"]
    x_test = h5py.File(googlefolder+'camelyonpatch_level_2_split_test_x.h5', 'r')["x"]
    y_test = h5py.File(googlefolder+'camelyonpatch_level_2_split_test_y.h5', 'r')["y"]
    x_train = h5py.File(googlefolder+'camelyonpatch_level_2_split_train_x.h5', 'r')["x"]

        # x_test = h5py.File('camelyonpatch_level_2_split_test_x.h5', 'r')["x"]
        # y_test = h5py.File('camelyonpatch_level_2_split_test_y.h5', 'r')["y"]


        # meta_train = pd.read_csv('camelyonpatch_level_2_split_train_meta.csv')
        # meta_valid = pd.read_csv('camelyonpatch_level_2_split_valid_meta.csv')
        # meta_test = pd.read_csv('camelyonpatch_level_2_split_test_meta.csv')


    # if K.image_data_format() == 'channels_first':
    #     raise NotImplementedError()

    return (np.array(x_train), np.array(y_train)), (np.array(x_valid), np.array(y_valid)), (np.array(x_test), np.array(y_test))
    #return (np.array(x_test), np.array(y_test))

if __name__ == '__main__':
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data()
    #(x_test, y_test) = load_data()

    if not os.path.isdir("data/train/cancer/"):
        Path("data/train/cancer/").mkdir(parents=True, exist_ok=True)
    if not os.path.isdir("data/train/healthy/"):
        Path("data/train/healthy/").mkdir(parents=True, exist_ok=True)
    if not os.path.isdir("data/valid/cancer/"):
        Path("data/valid/cancer/").mkdir(parents=True, exist_ok=True)
    if not os.path.isdir("data/valid/healthy/"):
        Path("data/valid/healthy/").mkdir(parents=True, exist_ok=True)
    if not os.path.isdir("data/test/cancer/"):
        Path("data/test/cancer/").mkdir(parents=True, exist_ok=True)
    if not os.path.isdir("data/test/healthy/"):
        Path("data/test/healthy/").mkdir(parents=True, exist_ok=True)

    index = 0
    for x, y in zip(x_train, y_train):
        print(y[0][0][0])
        label = y[0][0][0]
        if label == 1:
            im = Image.fromarray(x)
            im.save(f"data/train/cancer/{index}.png")
            index += 1
        elif label == 0:
            im = Image.fromarray(x)
            im.save(f"data/train/healthy/{index}.png")
            index += 1

    index = 0
    for x, y in zip(x_valid, y_valid):
        print(y[0][0][0])
        label = y[0][0][0]
        if label == 1:
            im = Image.fromarray(x)
            im.save(f"data/valid/cancer/{index}.png")
            index += 1
        elif label == 0:
            im = Image.fromarray(x)
            im.save(f"data/valid/healthy/{index}.png")
            index += 1

    index = 0
    for x,y in zip(x_test,y_test):
        print(y[0][0][0])
        label = y[0][0][0]
        if label == 1:
            im = Image.fromarray(x)
            im.save(f"data/test/cancer/{index}.png")
            index+=1
        elif label == 0:
            im = Image.fromarray(x)
            im.save(f"data/test/healthy/{index}.png")
            index+=1




    # train_tensor_x = torch.Tensor(x_train)  # transform to torch tensor
    # train_tensor_y = torch.Tensor(y_train)

    # valid_tensor_x = torch.Tensor(x_valid)  # transform to torch tensor
    # valid_tensor_y = torch.Tensor(y_valid)
    #
    # train_dataset = TensorDataset(train_tensor_x, train_tensor_y)  # create your datset
    # train_dataloader = DataLoader(train_dataset)
    #
    # valid_dataset = TensorDataset(valid_tensor_x, valid_tensor_y)  # create your datset
    # valid_dataloader = DataLoader(valid_dataset)
    #
    # pretrained_resnet_34 = timm.create_model('resnet34', pretrained=True, num_classes=2)