import os
import csv
import argparse

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from tqdm import tqdm

from dataset import get_fold_from_image_folder, DataFolderPair
from GoogLeNet import GoogleNetTrainer

parser = argparse.ArgumentParser()

# Data
parser.add_argument("--data-path", "-d", type=str, help="The dataset root path")
parser.add_argument("--batch-size", "-b", type=int, help="batch size", default=4)
parser.add_argument('--workers', '-j', default=8, type=int,
                    help='number of data loading workers (default: 8)')
parser.add_argument("--epoch", "-e", type=int, default=25, help="The epochs of training")

# Model
parser.add_argument("--model", "-m", type=str, help="The model architecture", default="inception_v3")
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

    input_dataset = ImageFolder(data_dir, train_transforms)

    fold_train_test_dataset = get_fold_from_image_folder(input_dataset)

    for i, (train_dataset, test_dataset) in enumerate(tqdm(fold_train_test_dataset)):
        result = fold_train(train_dataset, test_dataset, i)


pass


def fold_train(train_dataset, test_dataset, fold):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    trainer = GoogleNetTrainer(dataloaders, dataset_sizes)
    checkpoint_epoch = 0
    if input_args.resume:
        print(f"Continue training from checkpoint {input_args.resume}")
        checkpoint = torch.load(input_args.resume)
        trainer.net.load_state_dict(checkpoint["model_state_dict"])
        trainer = GoogleNetTrainer(dataloaders, dataset_sizes)
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])

        checkpoint_epoch = checkpoint["epoch"]
    pass

    trained_model_ft, acc_loss_history, epoch = trainer.train(num_epochs=input_args.epoch,
                                                              previous_epoch=checkpoint_epoch,
                                                              save_model_name=f"GoogleNet-classification-fold{fold+1}.pth")
    return trained_model_ft, acc_loss_history, epoch


pass


if __name__ == '__main__':
    main()


