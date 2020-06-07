import os
import shutil
from torchvision import transforms
import torchvision
import torch
import glob
from PIL import Image
import numpy as np
#import cupy as np
from tqdm import tqdm

def make_dir(path, del_before=False):
    if del_before and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.mkdir(path)


def save_attack_img(imgs, fn, attack_method, model_name='resnet50', iterate_num=0, output_dir='test_out',
                    dataset='cifar10',device='cuda'):
    trans_image = transforms.Compose([
        transforms.ToPILImage(),
    ])
    if dataset == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, 32, 32)
        t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 32, 32)
    elif dataset == 'mnist':
        mean, std = (0.1307,), (0.3081,)
        t_mean = torch.FloatTensor(mean).view(1, 1, 1).expand(1, 28, 28)
        t_std = torch.FloatTensor(std).view(1, 1, 1).expand(1, 28, 28)
    elif dataset == 'imagenet':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, 224, 224)
        t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 224, 224)
    #t_mean,t_std = t_mean.to(device),t_std.to(device)
    # print(imgs)
    imgs=imgs.cpu()
    for i, image in enumerate(imgs):
        image = image * t_std + t_mean
        image = trans_image(image.cpu())
        if attack_method in ('mifgsm'):
            filename = '_'.join([fn[i], attack_method, model_name, str(iterate_num)]) + '.png'
        else:
            filename = '_'.join([fn[i], attack_method, model_name]) + '.png'
        image.save(os.path.join(output_dir, filename), format='PNG')


def get_dataset(dataset_name, is_train, dataset_dir='./dataset', transform=None):
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root=dataset_dir, download=True, train=is_train,
                                               transform=transform)
    elif dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST(root=dataset_dir, download=True, train=is_train,
                                             transform=transform)
    elif dataset_name == 'imagenet':
        path = os.path.join(dataset_dir,'imagenet','train' if is_train else 'val')
        dataset = torchvision.datasets.ImageFolder(path,transform=transform)
    return dataset


class OriDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, data_name, transform, dataset_dir):
        self.data_name = data_name
        self.dataset = get_dataset(dataset_name=dataset,is_train=data_name=='train',dataset_dir=dataset_dir,
                                   transform=transform)

    def __getitem__(self, index):
        img = self.dataset[index][0]
        label = self.dataset[index][1]
        fn = '_'.join([self.data_name, str(index)])
        return img, label, fn

    def __len__(self):
        return len(self.dataset)


def get_orig_dataloader(batch_size, dataset='cifar10', dataset_dir='./dataset', is_tran=False,get_train=True):
    transform = get_transform(dataset)
    if get_train:
        trainset = OriDataset(dataset=dataset, data_name='train',
                         transform=get_transform(dataset, is_tran=True) if is_tran else transform,
                         dataset_dir=dataset_dir)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testset = OriDataset(dataset=dataset, data_name='test', transform=transform, dataset_dir=dataset_dir)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return (trainloader, testloader) if get_train else testloader


def get_transform(dataset='cifar10', is_tran=False):
    mean, std = None, None
    cifar10_mean, cifar10_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    mnist_mean, mnist_std = (0.1307,), (0.3081,)
    imagenet_mean, imagenet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    crop = {'mnist': 28, 'cifar10': 32, 'imagenet': 224}
    if dataset == 'cifar10':
        mean, std = cifar10_mean, cifar10_std
    elif dataset == 'mnist':
        mean, std = mnist_mean, mnist_std
    elif dataset == 'imagenet':
        mean, std = imagenet_mean, imagenet_std
    if not is_tran:
        transform = transforms.Compose([
            transforms.Resize((crop[dataset],crop[dataset])),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((crop[dataset], crop[dataset])),
            transforms.RandomCrop(crop[dataset]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transform


class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, dataset, transform=None, root='./dataset', out_noise=False,select_category=-1):
        self.dataset = dataset
        #if dataset != 'imagenet':
        self.trainset = get_dataset(dataset, is_train=True)
        self.testset = get_dataset(dataset, is_train=False)
        #else:
        #    self.testset = get_dataset(dataset, is_train=False)
        imgs = [img_path for img_path in glob.glob(os.path.join(dataset_folder, '*.png'))]
        labels = [-1]*len(imgs)
        for i,img in enumerate(imgs):
            img_set, img_id = os.path.splitext(os.path.basename(img))[0].split('_')[:2]
            labels[i] = self.trainset.targets[int(img_id)] if img_set == 'train' else self.testset.targets[int(img_id)]
        if select_category==-1:
            self.imgs = imgs
            self.labels = labels
        else:
            length = labels.count(select_category)
            self.imgs = [''] * length
            self.labels= [select_category] * length
            l=0
            for i,img in enumerate(imgs):
                if labels[i] == select_category:
                    self.imgs[l]=img
                    l+=1
        self.transform = transform
        self.out_noise = out_noise


    def __getitem__(self, index):
        img_set, img_id = os.path.splitext(os.path.basename(self.imgs[index]))[0].split('_')[:2]
        img = Image.open(self.imgs[index]).convert('RGB' if self.dataset != 'mnist' else 'L')
        if self.out_noise:
            ori = self.trainset[int(img_id)][0] if img_set == 'train' else self.testset[int(img_id)][0]
            if self.dataset=='imagenet':
                ori = ori.resize((224,224),Image.ANTIALIAS)
            img = self.get_noise(img, ori)
        if self.transform:
            img = self.transform(img)

        return img, self.labels[index], os.path.basename(self.imgs[index])

    def get_noise(self, img, ori):
        noise = Image.fromarray(np.array(img) - np.array(ori))
        return noise

    def __len__(self):
        return len(self.imgs)


def get_folder_dataloader(dataset_folder, dataset='cifar10', batch_size=256, root='./dataset', out_noise=False,
                          select_category = -1):
    transform = get_transform(dataset=dataset)
    dataset = FolderDataset(dataset_folder, dataset=dataset, transform=transform, root=root, out_noise=out_noise,
                            select_category=select_category)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader
'''
class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, dataset='cifar10', transform=None, root='./dataset'):
        if dataset != 'imagenet':
            trainset = get_dataset(dataset,is_train=True)
            #testset = get_dataset(dataset,is_train=False)
        else:
            trainset = get_dataset(dataset,is_train=False)
        imgs_path = [img_path for img_path in glob.glob(os.path.join(dataset_folder, '*.png'))]
        imgs_info = [os.path.splitext(os.path.basename(ip))[0].split('_')[:2] for ip in imgs_path]
        try:
            with tqdm(imgs_path,ncols=64) as tq:
                self.imgs = [Image.open(x).convert('RGB' if dataset != 'mnist' else 'L') for x in tq]
        except KeyboardInterrupt:
            tq.close()
            raise
        tq.close()
        try:
            orig_imgs=[None]*len(trainset)
            orig_label = [0]*len(trainset)
            with tqdm(trainset, ncols=64) as tq:
                #self.imgs+=[t[0] for t in tq]  # + [t[0] for t in testset]
                for i,(img,label) in enumerate(tq):
                    orig_imgs[i]=img
                    orig_label[i]=label
        except KeyboardInterrupt:
            tq.close()
            raise
        tq.close()
        self.imgs+=orig_imgs
        print('imgs_done')
        self.labels = [orig_label[int(img_id)] for (img_set, img_id) in imgs_info] + orig_label
        print('labels done')
        self.fn = ['_'.join(info) for info in imgs_info] + ['_'.join(('train', str(i))) for i in
                                                            range(len(trainset))]  # \
        # +['_'.join(('test',str(i))) for i in range(len(testset))]
        self.transform = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        fn = self.fn[index]

        if self.transform:
            img = self.transform(img)
        return img, label, fn

    def __len__(self):
        return len(self.imgs)
'''

def multiple_dataloader(dataset_folder, batch_size=256, dataset='cifar10', is_tran=True, divid=True, root='./dataset'):
    transform = get_transform(dataset=dataset, is_tran=is_tran)
    adv_dataset = FolderDataset(dataset_folder=dataset_folder,dataset=dataset,root=root,out_noise=False,
                                transform=transform)
    ori_dataset = OriDataset(dataset=dataset,data_name='True' if is_tran==True else 'False',
                                        transform=transform,dataset_dir=root)
    mul_dataset = torch.utils.data.ConcatDataset((ori_dataset,adv_dataset))
    if divid:
        train_size = int(0.8 * len(mul_dataset))
        test_size = len(mul_dataset) - train_size
        train_set, test_set = torch.utils.data.random_split(mul_dataset, [train_size, test_size])
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        return trainloader, testloader
    else:
        dataloader = torch.utils.data.DataLoader(mul_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return dataloader


class BlankNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, data_name, transform, dataset_dir):
        self.data_name = data_name
        self.dataset = torchvision.datasets.CIFAR10(root=dataset_dir, download=True, train=(data_name == 'train'),
                                                    transform=transform)

    def __getitem__(self, index):
        img = self.dataset[index][0]
        label = self.dataset[index][1]
        fn = '_'.join([self.data_name, str(index)])
        return img, label, fn

    def __len__(self):
        return len(self.dataset)

