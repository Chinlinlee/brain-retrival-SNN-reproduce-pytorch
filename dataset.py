# From: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# Github remote repo: https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import os
import random
from copy import copy
import glob
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union

import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchsampler import ImbalancedDatasetSampler
from scipy.io import loadmat
import h5py
import numpy as np
from sklearn.model_selection import KFold


def get_fold_from_image_folder(i_dataset, fold_num=5):
    train_test_dataset_list = []
    kfold_obj = KFold(n_splits=fold_num, shuffle=True, random_state=100)
    for index, (train_index, test_index) in enumerate(kfold_obj.split(i_dataset)):
        dataset_train = torch.utils.data.Subset(i_dataset, train_index)
        dataset_test = torch.utils.data.Subset(i_dataset, test_index)
        train_test_dataset_list.append([dataset_train, dataset_test])
    pass
    return train_test_dataset_list


pass


class DataFolderPair(ImageFolder):
    def __init__(self, **kwargs):
        super(DataFolderPair, self).__init__(**kwargs)

    pass

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample1, sample2, yp) where yp is 0 to 1 that mean two sample is same label or not
        """
        path, target = self.samples[index]

        # Chose either positive(0) or negative(1)
        self.yp = np.random.randint(0, 2)
        new_imgs = list(self.imgs)
        new_imgs.remove(self.imgs[index])
        length = len(new_imgs)

        np_random = np.random.RandomState(42)

        # When yp = 1 that choose the negative image, positive image otherwise
        img2 = None
        if self.yp == 1:
            label2 = target
            while label2 == target:
                choice = np_random.choice(length)
                img2, label2 = new_imgs[choice]
            pass
        else:
            label2 = target + 1
            while label2 != target:
                choice = np_random.choice(length)
                img2, label2 = new_imgs[choice]
            pass
        pass

        sample = Image.open(path).convert('RGB')
        sample2 = Image.open(img2).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
            sample2 = self.transform(sample2)

        return sample, sample2, self.yp

    pass


pass


class GuangzhouBrainDataset(Dataset):
    def __init__(self, root, folds, transform=None):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.transform = transform
        current_folder = os.path.dirname(os.path.abspath(__file__))
        cvind_file = h5py.File(os.path.join(current_folder, "cvind5fold.mat"))
        labels_file = loadmat(os.path.join(current_folder, "label.mat"))
        labels = np.array(labels_file["label"]).flatten()
        labels = labels - 1
        cvind = [int(v) for v in cvind_file["cvind"][0]]
        imgs = glob.glob(os.path.join(root, "**/*.jpg"), recursive=True)
        self.samples = []
        for fold in folds:
            fold_img_index = [i for i, v in enumerate(cvind) if v == fold]
            for img in imgs:
                img_index = int(os.path.basename(img).split(".")[0]) - 1
                img_fullname = os.path.abspath(img)
                if img_index in fold_img_index:
                    self.samples.append([img_fullname, int(labels[img_index])])
                pass
            pass
        pass
        self.classes = ["Meningioma", "Glioma", "Pituitary"]
        self.imgs = self.samples

    pass

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        path, target = self.imgs[index]
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    pass

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.samples)

    pass


pass


class GuangzhouBrainDatasetPair(GuangzhouBrainDataset):
    def __init__(self, root, folds, transform=None):
        super(GuangzhouBrainDatasetPair, self).__init__(root, folds, transform)

    pass

    def __getitem__(self, index):
        # from https://github.com/HarshSulakhe/siamesenetworks-pytorch/blob/master/dataset.py
        path, target = self.imgs[index]

        # Chose either positive(0) or negative(1)
        self.yp = np.random.randint(0, 2)
        new_imgs = list(self.imgs)
        new_imgs.remove(self.imgs[index])
        length = len(new_imgs)

        np_random = np.random.RandomState(42)
        img2 = None
        if self.yp == 1:
            label2 = target
            while label2 == target:
                choice = np_random.choice(length)
                img2, label2 = new_imgs[choice]
            pass
        else:
            label2 = target + 1
            while label2 != target:
                choice = np_random.choice(length)
                img2, label2 = new_imgs[choice]
            pass
        pass

        sample = Image.open(path).convert('RGB')
        sample2 = Image.open(img2).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
            sample2 = self.transform(sample2)

        return sample, sample2, self.yp

    pass


pass


def get_five_fold_dataset(data_root):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset_fold1 = GuangzhouBrainDataset(data_root, [2, 3, 4, 5], train_transforms)
    test_dataset_fold1 = GuangzhouBrainDataset(data_root, [1], test_transforms)

    train_dataset_fold2 = GuangzhouBrainDataset(data_root, [1, 3, 4, 5], train_transforms)
    test_dataset_fold2 = GuangzhouBrainDataset(data_root, [2], test_transforms)

    train_dataset_fold3 = GuangzhouBrainDataset(data_root, [1, 2, 4, 5], train_transforms)
    test_dataset_fold3 = GuangzhouBrainDataset(data_root, [3], test_transforms)

    train_dataset_fold4 = GuangzhouBrainDataset(data_root, [1, 2, 3, 5], train_transforms)
    test_dataset_fold4 = GuangzhouBrainDataset(data_root, [4], test_transforms)

    train_dataset_fold5 = GuangzhouBrainDataset(data_root, [1, 2, 3, 4], train_transforms)
    test_dataset_fold5 = GuangzhouBrainDataset(data_root, [5], test_transforms)

    class_names = train_dataset_fold1.classes

    train_test_datasets = [[train_dataset_fold1, test_dataset_fold1],
                           [train_dataset_fold2, test_dataset_fold2],
                           [train_dataset_fold3, test_dataset_fold3],
                           [train_dataset_fold4, test_dataset_fold4],
                           [train_dataset_fold5, test_dataset_fold5]]
    return train_test_datasets, class_names


pass

