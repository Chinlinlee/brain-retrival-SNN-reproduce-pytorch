import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

from dataset import get_fold_from_image_folder

parser = argparse.ArgumentParser()

parser.add_argument("--data-path", "-d", type=str, help="The dataset folder", required=True)
parser.add_argument("--weight", "-w", type=str, help="The pth file of model weight", default="inception-v3-brain.pth")
parser.add_argument("--fold", type=int, default=1)
input_args = parser.parse_args()


def do_test_accuracy(net, i_testloader):
    net.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in i_testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {len(i_testloader.dataset)} test images: {100 * correct // total} %')


pass


def do_test_accuracy_categories(net, i_testloader):
    net.eval()
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in class_names}
    total_pred = {classname: 0 for classname in class_names}
    # again no gradients needed
    with torch.no_grad():
        for data in i_testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[class_names[label]] += 1
                pass
                total_pred[class_names[label]] += 1
            pass
        pass
    pass
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    pass


pass

if __name__ == '__main__':
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

    input_dataset = ImageFolder(data_dir, test_transforms)
    fold_train_test_dataset = get_fold_from_image_folder(input_dataset)

    if isinstance(fold_train_test_dataset[0][0].dataset, torch.utils.data.Subset):
        class_names = fold_train_test_dataset[0][0].dataset.dataset.classes
        class_num = len(class_names)
    else:
        class_names = fold_train_test_dataset[0][0].dataset.classes
        class_num = len(class_names)
    pass

    model_googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    num_ftrs = model_googlenet.fc.in_features
    model_googlenet.fc = nn.Linear(num_ftrs, class_num)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weight = torch.load(input_args.weight)
    if "model_state_dict" in weight:
        model_googlenet.load_state_dict(weight["model_state_dict"])
    else:
        model_googlenet.load_state_dict(weight)
    pass

    print(f"test for fold {input_args.fold}")
    train_test_dataset = fold_train_test_dataset[input_args.fold-1]
    test_dataset = train_test_dataset[1]
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0)
    do_test_accuracy(model_googlenet, test_dataloader)
    do_test_accuracy_categories(model_googlenet, test_dataloader)

pass

