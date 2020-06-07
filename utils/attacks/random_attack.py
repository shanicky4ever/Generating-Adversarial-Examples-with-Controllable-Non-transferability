import torch
from torch import nn,optim
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

def random_attack(dataloader,max_noise,h=32,w=32,c=3,output_dir='test_out'):
    trans=transforms.ToTensor()
    try:
        with tqdm(dataloader) as tq:
            for i,(imgs,labels,fn) in enumerate(tq):
                for j,img in enumerate(imgs):
                    noise=(np.random.rand(h,w,c)*(2*max_noise)-max_noise)*2.1
                    np.array(noise,dtype=np.float32)
                    noise=trans(noise)
                    img+=noise
                    save_attack_img([img],fn=[fn[j]],attack_method='random',model_name='uni',output_dir=output_dir)


    except KeyboardInterrupt:
        tq.close()
        raise
    tq.close()
