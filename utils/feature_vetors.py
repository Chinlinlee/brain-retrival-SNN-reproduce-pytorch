"""
From: https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/networks/imageretrievalnet.py
"""

import argparse
import torch
import numpy as np


def extract_ss(net, i_input):
    return net(i_input, emb=True).cpu().data.squeeze()


pass

def extract_snn_ss(net, i_input):
    return net.forward_once(i_input).cpu().data.squeeze()


pass


def extract_vectors(net, i_device, i_dataloader):
    net.cuda()
    net.eval()
    img_feats = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(i_dataloader):
            inputs = inputs.to(i_device)

            img_feats.append([i, labels.cpu().numpy()[0], extract_ss(net, inputs)])
        pass
    pass

    return img_feats


pass

def extract_snn_vectors(net, i_device, i_dataloader):
    net.cuda()
    net.eval()
    img_feats = []

    with torch.no_grad():
        for i, (input1, labels) in enumerate(i_dataloader):
            input1 = input1.to(i_device)

            img_feats.append([i, labels.cpu().numpy()[0], extract_snn_ss(net, input1)])
        pass
    pass

    return img_feats
pass


def extract_vectors_gem(net, i_device, i_dataloader):
    net.cuda()
    net.eval()
    img_feats = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(i_dataloader):
            inputs = inputs.to(i_device)
            feature = net(inputs, emb=True).cpu().data.squeeze()

            img_feats.append([i, labels.cpu().numpy()[0], feature])
        pass
    pass

    return img_feats


pass


def compute_distance_l2(x, y):
    temp = x - y
    dist = np.sqrt(np.dot(temp.T, temp))
    return dist


pass