import argparse
import math
import random
import sys

import scipy
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from torchvision.datasets import ImageFolder

from dataset import get_five_fold_dataset, get_fold_from_image_folder
from Siamese import SiameseNet
from utils.feature_vetors import extract_snn_vectors

parser = argparse.ArgumentParser()

parser.add_argument("--data-path", "-d", type=str, help="The dataset folder", required=True)
parser.add_argument("--stage1", type=str, required=True)
parser.add_argument("--weight", "-w", type=str, help="The pth file of model weight")
parser.add_argument("--seed", type=int, default=random.randint(1, 2 ** 32 - 1))
parser.add_argument("--fold", type=int, required=True, default=1)
input_args = parser.parse_args()


def get_specific_label_indices(i_label, dataloader):
    hit_indices = []
    for i, (data_path, label) in enumerate(dataloader):
        if label == i_label:
            hit_indices.append(i)
        pass
    pass

    return hit_indices


pass


def do_retrieval_eval_specific_label(indices, db_indices, i_q_vectors, i_db_vectors, k=10):
    precision_list = []
    precision_at_k_list = []

    for index in indices:
        q_vector_index, q_vector_label, q_vector_feature = i_q_vectors[index]

        # Get every db vector distance from q_vector
        distance_list = []
        for db_vector in i_db_vectors:
            db_vector_index, db_vector_label, db_vector_feature = db_vector

            distance = scipy.spatial.distance.cityblock(q_vector_feature, db_vector_feature)
            distance_list.append([db_vector_label, distance])
        pass

        # sort with distance
        sorted_distance_list = sorted(distance_list, key=lambda v: v[1])

        # Calc average precision
        relevant_count = 0
        precision = 0
        for i, rank_info in enumerate(sorted_distance_list):
            rank_info_label = rank_info[0]
            if rank_info_label == q_vector_label:
                relevant_count = relevant_count + 1
                precision = precision + (relevant_count / (i + 1))
            pass

            if (i + 1) == k:
                precision_at_k = relevant_count / k
                precision_at_k_list.append(precision_at_k / len(indices))
            pass
        pass
        precision_list.append(precision / len(db_indices))
    pass

    avg_precision_sum = np.array(precision_list).sum()
    precision_at_k_sum = np.array(precision_at_k_list).sum()
    return avg_precision_sum, precision_at_k_sum


pass

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(input_args.seed)
    torch.cuda.manual_seed(input_args.seed)
    np.random.seed(input_args.seed)
    random.seed(input_args.seed)

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

    input_dataset = ImageFolder(input_args.data_path, test_transforms)
    fold_train_test_dataset = get_fold_from_image_folder(input_dataset)

    if isinstance(fold_train_test_dataset[0][0].dataset, torch.utils.data.Subset):
        class_names = fold_train_test_dataset[0][0].dataset.dataset.classes
        class_num = len(class_names)
    else:
        class_names = fold_train_test_dataset[0][0].dataset.classes
        class_num = len(class_names)
    pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SiameseNet(torch.load(input_args.stage1), class_names)

    weight = torch.load(input_args.weight)
    if "model_state_dict" in weight:
        model.load_state_dict(weight["model_state_dict"])
    else:
        model.load_state_dict(weight)

    overall_map_list = []
    overall_precision_10_list = []

    train_test_dataset = fold_train_test_dataset[input_args.fold - 1]
    test_dataset = train_test_dataset[1]
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0)
    train_dataset = train_test_dataset[0]
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=0)
    q_vectors = extract_snn_vectors(model, device, test_dataloader)
    db_vectors = extract_snn_vectors(model, device, train_dataloader)

    avg_precision_list = np.zeros(class_num)
    for index, name in enumerate(class_names):
        class_train_indices = get_specific_label_indices(index, train_dataloader)
        class_test_indices = get_specific_label_indices(index, test_dataloader)

        avg_precision_sum, precision_at_10 = do_retrieval_eval_specific_label(
            class_test_indices,
            class_train_indices,
            q_vectors,
            db_vectors
        )
        class_avg_precision = avg_precision_sum / len(class_test_indices)
        print(len(class_test_indices))
        print(f"fold:{input_args.fold}, {name} mAP: {np.around(class_avg_precision*100, decimals=2)}, "
              f"precision@10: {np.around(precision_at_10 * 100, decimals=2)}")

        avg_precision_list[index] = avg_precision_sum
    pass

    print(len(test_dataset))
    mean_avg_precision = avg_precision_list.sum() / len(test_dataset)
    print(f"fold:{input_args.fold}, all mAP: {np.around(mean_avg_precision * 100, decimals=2)}")
    print(f"use seed: {input_args.seed}")

pass
