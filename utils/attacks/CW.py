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

class CW:
    def __init__(self,model_name,output_dir='test_out',device='cuda',max_iters=10,targeted=False,lr=5e-3,
                 abort_early=True,initial_const=0.5,largest_const=32,reduce_const=False,decrease_factor=0.9,
                 const_factor=2.0,num_classes=10):
        self.output_dir=output_dir
        self.model_name=model_name
        self.net = models.get_model(model_name, is_pretrained=True, device=device)
        self.device=device
        self.max_iter=max_iters
        self.targeted=targeted
        self.lr=lr
        self.abort_early=abort_early
        self.initial_const=initial_const
        self.largest_const=largest_const
        self.reduce_const=reduce_const
        self.decrease_factor=decrease_factor
        self.const_factor=const_factor
        self.num_classes=num_classes
        self.shape=(1,3,32,32)

    def compare(self,x,y):
        return x == y if self.targeted else x != y

    def tanh(self,x):
        return torch.nn.Tanh()(x)

    def get_max(self,x,y):
        return x.data if x.data>y else y

    def torch_arctanh(self,x,ep=1e-6):
        x*=(1-ep)
        return 0.5*torch.log((1+x)/(1-x))

    def grad_descent(self,ori_img,label,st_tmp,tt,const):
        con=const
        img=self.torch_arctanh(ori_img*1.999999)
        st=self.torch_arctanh(st_tmp*1.999999)
        img=img.unsqueeze(0)
        tau=tt
        timg,tlabel=img,label
        #tlabel=torch.zeros(1,self.num_classes).scatter_(1,tlabel,1)
        tlabel=torch.sparse.torch.eye(self.num_classes,device=self.device).index_select(0,label)
        #torch.unsqueeze(tlabel,1)
        #print(label_onehot)
        #tlabel=label_onehot.scatter(1,tlabel.unsqueeze(1),1.0)
        while con<self.largest_const:
            modifier=torch.zeros(self.shape,requires_grad=True,device=self.device)
            optimizer=optim.Adam([modifier],lr=self.lr)
            st_img=st.clone()
            for step in range(self.max_iter):
                new_img=self.tanh(modifier+st_img)/2
                output=self.net(new_img)
                ori_output=self.net(self.tanh(timg)/2)
                real=torch.mean(tlabel*output)
                other=torch.max((1-tlabel)*output-(tlabel*10000))
                loss1=torch.max(((other-real) if self.targeted else (real-other)),torch.zeros_like(real))
                loss2=torch.sum(torch.max(torch.abs(new_img-self.tanh(timg)/2)-tau,torch.zeros_like(new_img)))
                loss=con*loss1+loss2
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                if loss<0.0001*con and self.abort_early:
                    works=self.compare(torch.argmax(output),torch.argmax(tlabel))
                    if works:
                        return output,ori_output,new_img,con
            con=con*self.const_factor

    def attack_single(self,img,label):
        st_img=img
        tau=0.1
        con=self.initial_const
        while tau>1/256:
            res=self.grad_descent(img.clone(),label,st_img.clone(),tau,con)
            if not res:
                return st_img
            scores,origin_scores,nimg,con=res
            if self.reduce_const:
                con=torch.div(con,2)
            actualtau=torch.max(torch.abs(nimg-img))
            if actualtau<tau:
                tau=actualtau
            st_img=nimg.clone()
            tau=torch.mul(tau,self.decrease_factor)
        return st_img

    def attack(self,dataloader):
        fail=0
        try:
            with tqdm(total=len(dataloader.dataset)) as tq:
                for i,(img,label,fn) in enumerate(dataloader):
                    #result=[]
                    for j in range(len(img)):
                        x,y=img[j].to(self.device),label[j].to(self.device)
                        res=self.attack_single(x,y)
                       # result.append(res)
                        if len(res.shape)==3:
                            res=res.unsqueeze(0)
                            fail+=1
                        save_attack_img(res,fn=[fn[j]],attack_method='CW',
                                        model_name=self.model_name,output_dir=self.output_dir)
                        tq.update(1)
        except KeyboardInterrupt:
            tq.close()
            raise
        tq.close()
        print(fail)
