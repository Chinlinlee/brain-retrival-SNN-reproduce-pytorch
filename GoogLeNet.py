import math

import torch
from pytorch_metric_learning import distances, losses
from torch import nn, optim
from torch.nn import Parameter
from torch.optim import lr_scheduler
from torchvision import models
from torch.nn import functional as F
from tqdm import tqdm
import time
import os
import copy
from enum import Enum
from functools import partial


class GoogleNetTrainer:
    def __init__(self, dataloaders, dataset_sizes):
        model_googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        num_ftrs = model_googlenet.fc.in_features

        if isinstance(dataloaders["train"].dataset, torch.utils.data.Subset):
            class_num = len(dataloaders["train"].dataset.dataset.classes)
        else:
            class_num = len(dataloaders["train"].dataset.classes)
        pass

        model_googlenet.fc = nn.Linear(num_ftrs, class_num)
        self.net = model_googlenet

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.acc_loss_history = []

        learning_rate = 3e-4
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        exp_decay = math.exp(-0.01)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=exp_decay)

        self.criterion = nn.CrossEntropyLoss()

    pass

    def train(self, num_epochs=25, previous_epoch=0, save_model_name=None, need_val=False):
        since = time.time()

        best_model_wts = copy.deepcopy(self.net.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            self.net.train()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(self.dataloaders["train"]):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = self.net(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    loss.backward()
                    self.optimizer.step()
                pass
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            pass
            self.scheduler.step()

            epoch_loss = running_loss / self.dataset_sizes["train"]
            epoch_acc = running_corrects.double() / self.dataset_sizes["train"]

            print(f'train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            self.acc_loss_history.append([epoch + previous_epoch, "train", epoch_loss, f"{epoch_acc:.4f}"])

            if need_val:
                best_acc, best_model_wts = self.val(epoch, best_acc, best_model_wts)
            pass
        pass

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best Acc: {best_acc:4f}')

        if save_model_name is not None:
            self.save_model(num_epochs, save_model_name)
        pass

        return self.net, self.acc_loss_history, num_epochs + previous_epoch

    pass

    def val(self, current_epoch, i_best_acc, i_best_model_wts):
        self.net.eval()  # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(self.dataloaders["train"]):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = self.net(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
            pass
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        pass

        epoch_loss = running_loss / self.dataset_sizes["val"]
        epoch_acc = running_corrects.double() / self.dataset_sizes["val"]

        print(f'val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        self.acc_loss_history.append([current_epoch, "val", epoch_loss, f"{epoch_acc:.4f}"])

        # deep copy the model
        if epoch_acc > i_best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(self.net.state_dict())
            return best_acc, best_model_wts
        pass

        return i_best_acc, i_best_model_wts

    pass

    def save_model(self, epoch, filename):
        model_state_dict = self.net.state_dict()
        checkpoint = {
            "model_state_dict": model_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "loss": self.criterion.state_dict(),
            "epoch": epoch + 1
        }
        torch.save(checkpoint, filename)

    pass


pass
