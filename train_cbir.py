import os
import csv
import argparse

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import GuangzhouBrainDatasetPair
from Siamese import SiameseTrainer

parser = argparse.ArgumentParser()

# Data
parser.add_argument("--data-path", "-d", type=str, help="The dataset root path")
parser.add_argument("--batch-size", "-b", type=int, help="batch size", default=4)
parser.add_argument('--workers', '-j', default=8, type=int,
                    help='number of data loading workers (default: 8)')
parser.add_argument("--iteration", "-i", type=int, default=600, help="The iterations of training")
parser.add_argument("--stage1", type=str, required=True)

parser.add_argument("--resume", type=str, help="name of the latest checkpoint (default: None)")

input_args = parser.parse_args()


def main():
    data_dir = input_args.data_path
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

    train_test_datasets = []
    for index in np.arange(5):
        fold_indices = np.arange(5) + 1
        train_indices = np.delete(fold_indices, index)
        test_index = [index + 1]

        train_dataset_fold = GuangzhouBrainDatasetPair(data_dir, train_indices, train_transforms)
        test_dataset_fold = GuangzhouBrainDatasetPair(data_dir, test_index, test_transforms)

        train_test_datasets.append([train_dataset_fold, test_dataset_fold])
    pass

    for i, train_test_dataset in enumerate(tqdm(train_test_datasets)):
        result = fold_train(train_test_dataset[0], train_test_dataset[1], i)
    pass


pass


def fold_train(train_dataset, test_dataset, fold):
    dataloaders = {
        "train": DataLoader(train_dataset,
                            batch_size=input_args.batch_size,
                            shuffle=True,
                            num_workers=input_args.workers,
                            drop_last=True),
        "val": DataLoader(test_dataset,
                          batch_size=input_args.batch_size,
                          shuffle=False,
                          num_workers=input_args.workers),
    }
    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(test_dataset)
    }
    trainer = SiameseTrainer(dataloaders, dataset_sizes, torch.load(input_args.stage1))
    checkpoint_iteration = 0
    if input_args.resume:
        print(f"Continue training from checkpoint {input_args.resume}")
        checkpoint = torch.load(input_args.resume)
        trainer.net.load_state_dict(checkpoint["model_state_dict"])
        trainer = SiameseTrainer(dataloaders, dataset_sizes)
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])

        checkpoint_iteration = checkpoint["iteration"]
    pass

    trained_model_ft, acc_loss_history, iteration = trainer.train(iteration=input_args.iteration,
                                                                  previous_iteration=checkpoint_iteration,
                                                                  save_model_name=f"GoogleNet-CBIR-fold{fold + 1}.pth")
    return trained_model_ft, acc_loss_history, iteration


pass

if __name__ == '__main__':
    main()
