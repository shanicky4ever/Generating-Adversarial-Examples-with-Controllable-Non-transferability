import torchvision
from torchvision import transforms
import os
import csv
import torch
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
#import cupy as np
import torchvision
import shutil
import pickle
from tqdm import tqdm
from utils import data
import h5py


class NoiseToAdvDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, transform=None, dataset='cifar10'):
        print('get dataset')
        if dataset != 'imagenet':
            self.trainset = data.get_dataset(dataset, is_train=True)
            self.testset = data.get_dataset(dataset, is_train=False)
        else:
            self.testset = data.get_dataset(dataset, is_train=False)
        self.dataset = dataset
        self.transform = transform
        self.imgs = [img_path for img_path in glob.glob(os.path.join(dataset_folder, '*.png'))]

    def get_adv(self, img, ori):
        if self.dataset=='imagenet':
            ori=ori.resize((224,224),Image.ANTIALIAS)
        if self.transform:
            img = self.transform(img)
            ori = self.transform(ori)
        adv = img+ori
        return adv

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB' if self.dataset != 'mnist' else 'L')
        set_name, img_number = os.path.splitext(os.path.basename(self.imgs[index]))[0].split('_')[:2]
        img_number = int(img_number)
        ori = self.trainset[img_number][0] if set_name == 'train' else self.testset[img_number][0]
        label = self.trainset[img_number][1] if set_name == 'train' else self.testset[img_number][1]
        adv = self.get_adv(img, ori)
        return adv, label, os.path.splitext(os.path.basename(self.imgs[index]))[0]

    def __len__(self):
        return len(self.imgs)


def noise_to_adv_loader(dataset_folder, batch_size=256, dataset='cifar10'):
    transform = data.get_transform(dataset)
    dataset = NoiseToAdvDataset(dataset_folder=dataset_folder, transform=transform, dataset=dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader


class QuadDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, dataset='cifar10', white_model='resnet50', black_model=('resnet50', 'vgg16bn'),
                 transform=None, select_category=0, quad=(1,2,3,4), quad_multi=(1, 1, 1, 1)):
        self.dataset = dataset
        self.trainset = data.get_dataset(dataset, is_train=True)
        self.testset = data.get_dataset(dataset, is_train=False)
        self.transform = transform
        with open(os.path.join(dataset_folder, 'succ.pkl'), 'rb') as f:
            info = pickle.load(f)
        self.imgs = []
        self.labels = []
        stat = [0] * 4
        [bm1, bm2] = black_model
        try:
            flag_quad = {2: 1, 3: 2, 4: 4, 5: 3}
            with tqdm(info.items(), ncols=64) as tq:
                for img_name, pred in tq:
                    set_name, img_number = img_name.split('_')[:2]
                    img_number = int(img_number)
                    lab = self.trainset.targets[img_number] if set_name == 'train' \
                        else self.testset.targets[img_number]
                    if 0 <= select_category != lab:
                        continue
                    img_path = os.path.join(dataset_folder, img_name)
                    flag = (1 if pred[bm1] else 0) + (4 if pred[bm2] else 2)
                    '''
                        flag=2 not pred1 and not pred2 --> quad 1
                        flag=3     pred1 and not pred2 --> quad 2 
                        flag=4 not pred1 and     pred2 --> quad 4
                        flag=5     pred1 and     pred2 --> quad 3
                    '''
                    q = flag_quad[flag]
                    if q in quad:
                        self.labels += [quad.index(q)] * quad_multi[q - 1]
                        self.imgs += [img_path] * quad_multi[q - 1]
                        stat[q - 1] += 1
        except KeyboardInterrupt:
            tq.close()
            raise
        tq.close()
        print(stat)

    def get_noise(self, img, ori):
        if self.dataset=='imagenet':
            ori=ori.resize((224,224),Image.ANTIALIAS)
        if self.transform:
            ori = self.transform(ori)
            img = self.transform(img)
        noise = img-ori
        return noise

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB' if self.dataset != 'mnist' else 'L')
        set_name, img_number = os.path.splitext(os.path.basename(self.imgs[index]))[0].split('_')[:2]
        img_number = int(img_number)
        ori = self.trainset[img_number][0] if set_name == 'train' else self.testset[img_number][0]
        label = self.labels[index]
        noise = self.get_noise(img, ori)
        return noise, label, os.path.splitext(os.path.basename(self.imgs[index]))[0]

    def __len__(self):
        return len(self.imgs)


def get_quad_loader(dataset_folder, dataset='cifar10', batch_size=256, select_category=0, white_model='resnet50',
                    black_model=('vgg16bn', 'densenet121'), divid=True, quad=(1, 4), quad_multi=(1, 1, 1, 1)):
    transform = data.get_transform(dataset)
    dataset = QuadDataset(dataset_folder=dataset_folder, dataset=dataset, transform=transform,
                          select_category=select_category, white_model=white_model, black_model=black_model,
                          quad=quad, quad_multi=quad_multi)
    if divid:
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        return trainloader, testloader
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return dataloader


class NoiseDividDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, classifier, select_category, black_model,batch_size=256, transform=None,
                 device='cuda',tmp_folder='tmp', dataset='cifar10'):
        self.dataset = dataset
        self.transform = transform
        if dataset != 'imagenet':
            self.trainset = data.get_dataset(dataset, is_train=True)
            self.testset = data.get_dataset(dataset, is_train=False)
        else:
            self.testset = data.get_dataset(dataset, is_train=False)
        self.tmp_folder = tmp_folder
        self.imgs = []
        self.labels = []
        folder_loader = data.get_folder_dataloader(dataset_folder=dataset_folder, batch_size=batch_size,
                                        out_noise=False, dataset=dataset,select_category=select_category)
        try:
            with tqdm(folder_loader, ncols=64) as tq:
                for i, (imgs, labels, fns) in enumerate(tq):
                    fn = np.array(fns)
                    x = imgs.to(device)
                    y = labels.to(device)
                    o1 = black_model[0](x)
                    o2 = black_model[1](x)
                    _, p1 = o1.data.max(1)
                    _, p2 = o2.data.max(1)
                    pr=self.quad(p1==y,p2==y)
                    pick = [i for i, p in enumerate(pr) if y[i] == select_category]
                    for p in pick:
                        path = os.path.join(tmp_folder, fn[p])
                        self.imgs.append(path)
                        shutil.copyfile(os.path.join(dataset_folder, fns[p]), path)
                    self.labels += [pr[p] for p in pick]
        except KeyboardInterrupt:
            tq.close()
            raise
        tq.close()
        print([self.labels.count(i) for i in range(4)])

    def quad(self,a,b):
        res={2:0,3:1,4:3,5:2}
        quad=[0]*len(a)
        for i in range(len(a)):
            l=0
            l+=0 if a[i] else 1
            l+=2 if b[i] else 4
            quad[i]=res[l]
        return quad

    def get_noise(self, img, ori):
        if self.dataset == 'imagenet':
            ori = ori.resize((224,224),Image.ANTIALIAS)
        if self.transform:
            ori = self.transform(ori)
            img = self.transform(img)
        noise = img-ori

        return noise

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB' if self.dataset != 'mnist' else 'L')
        set_name, img_number = os.path.splitext(os.path.basename(self.imgs[index]))[0].split('_')[:2]
        img_number = int(img_number)
        ori = self.trainset[img_number][0] if set_name == 'train' else self.testset[img_number][0]
        noise = self.get_noise(img, ori)
        return noise, self.labels[index], os.path.basename(self.imgs[index])

    def __len__(self):
        return len(self.imgs)


def get_noise_divid_loader(dataset_folder, classifier, select_category, black_model,batch_size=256, device='cuda',
                           tmp_folder='tmp',dataset='cifar10'):
    transform = data.get_transform(dataset)
    dataset = NoiseDividDataset(dataset_folder=dataset_folder, classifier=classifier,
                                select_category=select_category,
                                batch_size=batch_size, transform=transform, device=device, tmp_folder=tmp_folder,
                                dataset=dataset,black_model=black_model)
    print(dataset.__len__())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader
