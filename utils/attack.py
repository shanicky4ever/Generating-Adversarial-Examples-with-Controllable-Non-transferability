import torch
from torch import nn, optim
import torchvision
from torchvision import transforms
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
from utils.attacks import fgsm,CW,random_attack


def i_fgsm(dataloader, model, model_name, max_iter=5, eps=1 / 8, alpha=1 / 12,
            device='cuda',output_dir='test_out', save_every_step=True,dataset='cifar10'):
    acc=fgsm.i_fgsm(dataloader, model,model_name,max_iter=max_iter, eps=eps, alpha=alpha,
                    device=device,output_dir=output_dir, save_every_step=save_every_step,
                     dataset=dataset)
    return acc

def i_fgsm_target(dataloader, net, net_name, target, num_classes=10,max_iter=5, eps=1 / 8, alpha=1 / 12,
                  device='cuda',output_dir='test_out', save_every_step=False,dataset='cifar10'):
    acc = fgsm.i_fgsm_target(dataloader, net, net_name, target, num_classes=num_classes,max_iter=max_iter, eps=eps,
                             alpha=alpha,device=device,output_dir=output_dir, save_every_step=save_every_step,
                             dataset=dataset)
    return acc


def CW_attack(dataloader,model_name,output_dir='test_out',device='cuda',max_iters=5,targeted=False,lr=5e-3,
              abort_early=True,initial_const=0.5,largest_const=20,reduce_const=False,decrease_factor=0.9,
              const_factor=2.0,num_classes=10):
    CW_solver=CW.CW(model_name,output_dir=output_dir)
    CW_solver.attack(dataloader)
    del CW_solver

def random_att(dataloader,model_name='uni',output_dir='test_out',max_noise=1/16,h=32,w=32,c=3):
    random_attack.random_attack(dataloader,max_noise=max_noise,output_dir=output_dir,h=h,w=w,c=c)