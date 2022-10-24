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
from contrative_loss import ContrastiveLoss
from torchsummary import summary


class SiameseNet(nn.Module):
    def __init__(self, google_net_weight, classes):
        super(SiameseNet, self).__init__()
        model_googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        num_ftrs = model_googlenet.fc.in_features
        model_googlenet.fc = nn.Linear(num_ftrs, len(classes))

        if "model_state_dict" in google_net_weight:
            model_googlenet.load_state_dict(google_net_weight["model_state_dict"])
        else:
            model_googlenet.load_state_dict(google_net_weight)
        pass

        backbone_list = list(model_googlenet.children())[:-2]
        self.google_features = nn.Sequential(*backbone_list)

        self.features = nn.Sequential(
            nn.Linear(num_ftrs, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 2)
        )

    pass

    def forward_once(self, x):
        output = self.google_features(x)
        output = output.view(-1, 1024)
        output = self.features(output)
        return output

    pass

    def forward(self, input1, input2):
        input1_features = self.forward_once(input1)
        input2_features = self.forward_once(input2)
        return input1_features, input2_features

    pass


pass


class SiameseTrainer:
    def __init__(self, dataloaders, dataset_sizes, google_net_weight):
        if isinstance(dataloaders["train"].dataset, torch.utils.data.Subset):
            class_names = dataloaders["train"].dataset.dataset.classes
        else:
            class_names = dataloaders["train"].dataset.classes
        pass

        self.net = SiameseNet(google_net_weight, class_names)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.acc_loss_history = []

        learning_rate = 3e-4
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        exp_decay = math.exp(-0.01)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=exp_decay)

        self.criterion = ContrastiveLoss(margin=0.2)

    pass

    def train(self, iteration, previous_iteration, save_model_name=None):
        since = time.time()

        run_iteration = 0
        stop = False

        batch_size = len(self.dataloaders["train"])

        while True:
            self.net.train()

            running_loss = 0.0

            for i, (input1, input2, labels) in enumerate(tqdm(self.dataloaders["train"])):
                input1 = input1.to(self.device)
                input2 = input2.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    output1, output2 = self.net(input1, input2)

                    loss = self.criterion(output1, output2, labels)

                    # backward + optimize only if in training phase
                    loss.backward()
                    self.optimizer.step()
                pass
                # statistics
                current_loss = loss.item()
                running_loss += loss.item()
                self.acc_loss_history.append([iteration + previous_iteration, "train", current_loss])
                run_iteration += 1

                if run_iteration >= iteration:
                    stop = True
                    break
                pass
            pass
            iteration_loss = running_loss / self.dataset_sizes["train"]
            print(f'[epoch: {int(run_iteration / batch_size)}] train Loss: {iteration_loss:.12f}')
            self.scheduler.step()

            if stop:
                break
            pass

        pass

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        if save_model_name is not None:
            self.save_model(iteration, save_model_name)
        pass

        return self.net, self.acc_loss_history, iteration + previous_iteration

    pass

    def save_model(self, iteration, filename):
        model_state_dict = self.net.state_dict()
        checkpoint = {
            "model_state_dict": model_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "loss": self.criterion.state_dict(),
            "iteration": iteration + 1
        }
        torch.save(checkpoint, filename)
    pass


pass

