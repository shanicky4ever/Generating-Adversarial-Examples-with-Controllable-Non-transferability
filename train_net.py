import torch
from torch import nn, optim
import torchvision
from torchvision import transforms
from tqdm import tqdm
from time import sleep
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import argparse
import os
from utils import data, models, train, common

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50')
    parser.add_argument('--pth_path', default='gray_net/state_dicts')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--resume', default='False')
    parser.add_argument('--pretrained', default='False')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--gpu_id', default='0')
    args = parser.parse_args()

    data.make_dir(args.pth_path, del_before=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = common.get_device()

    resume = args.resume == 'True'
    pth = os.path.join(args.pth_path, 'my_{}_{}.pt'.format(args.model, args.dataset))
    train_loader, test_loader = data.get_orig_dataloader(batch_size=args.batch_size, dataset=args.dataset,
                                                             is_tran=True, get_train=True)
    net = models.get_model(args.model, dataset=args.dataset, 
                            is_pretrained=args.pretrained == 'True', device=device)
    net.train()
    if resume:
        net.load_state_dict(torch.load(pth, map_location=device))
    sch_step=[80,160]

    acc = train.train(net, trainloader=train_loader, testloader=test_loader, epochs=args.epochs, lr=args.lr,
                        weight_decay=args.weight_decay,pth_path=pth,sch_step=sch_step)

