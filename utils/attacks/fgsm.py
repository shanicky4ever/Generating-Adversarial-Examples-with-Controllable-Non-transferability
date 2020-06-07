import torch
from torch import nn, optim
import torchvision
from torchvision import transforms
from models import *
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
import os
import shutil
import numpy as np
import pickle
import math
import random
from utils.data import save_attack_img
from utils import models


def i_fgsm(dataloader, net, net_name, max_iter=5, eps=1 / 8, alpha=1 / 12, device='cuda',
            output_dir='test_out', save_every_step=True,dataset='cifar10'):
    loss_fn = nn.CrossEntropyLoss()
    net.eval()
    correct = 0
    try:
        with tqdm(dataloader,ncols=64) as tq:
            for i, (imgs, labels, fn) in enumerate(tq):
                x, y = imgs.to(device), labels.to(device)
                x_var= Variable(x, requires_grad=True)
                for iter in range(max_iter):
                    zero_gradients(x_var)
                    outputs = net(x_var)
                    loss = loss_fn(outputs, y)
                    loss.backward()
                    grad = x_var.grad.data
                    x_grad = alpha * torch.sign(grad)
                    x_grad = torch.clamp(x_grad, -1 * eps, eps)
                    x_adv = torch.clamp(x_var + x_grad, -2.2, 2.2)
                    x_var.data = x_adv
                    if save_every_step:
                        save_attack_img(x_adv, fn, 'ifgsm', net_name, iter + 1, output_dir=output_dir,dataset=dataset)
                if not save_every_step:
                    save_attack_img(x_adv, fn, 'ifgsm', net_name, iter + 1, output_dir=output_dir,dataset=dataset)
                _, preds = net(x_var).data.max(1)
                correct += (preds != y).sum()
    except KeyboardInterrupt:
        tq.close()
        raise
    tq.close()

    acc = correct.cpu().numpy() / len(dataloader.dataset)
    return acc

def i_fgsm_target(dataloader, net, net_name, target, num_classes=10,max_iter=5, eps=1 / 8, alpha=1 / 12,
                  device='cuda',output_dir='test_out', save_every_step=False,dataset='cifar10'):
    loss_fn = nn.CrossEntropyLoss()
    net.eval()
    correct = 0
    try:
        with tqdm(dataloader,ncols=64) as tq:
            for i,(imgs,labels,fns) in enumerate(tq):
                x = imgs.to(device)
                y = torch.LongTensor([target]*len(labels)).to(device)
                x_var = Variable(x, requires_grad=True)
                for iter in range(max_iter):
                    zero_gradients(x_var)
                    outputs = net(x_var)
                    loss = -loss_fn(outputs, y)
                    loss.backward()
                    grad = x_var.grad.data
                    x_grad = alpha * torch.sign(grad)
                    x_grad = torch.clamp(x_grad, -1 * eps, eps)
                    x_adv = torch.clamp(x_var + x_grad, -1*eps, eps)
                    x_var.data = x_adv
                    if save_every_step:
                        save_attack_img(x_adv, fns, 'ifgsm', net_name, iter + 1, output_dir=output_dir,dataset=dataset)
                if not save_every_step:
                    save_attack_img(x_adv, fns, 'ifgsm', net_name, iter + 1, output_dir=output_dir,dataset=dataset)
                _, preds = net(x_var).data.max(1)
                correct += (preds == y).sum()
    except KeyboardInterrupt:
        tq.close()
        raise
    tq.close()
    acc = correct.cpu().numpy() / len(dataloader.dataset)
    return acc