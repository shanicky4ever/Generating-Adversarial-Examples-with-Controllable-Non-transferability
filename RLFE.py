import torch
from torchvision import transforms, datasets
import argparse
import os
from utils import data, common, models, train
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from gray_net import gray_net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nets', default=['resnet50','vgg16bn','densenet121'], nargs='+',
                        help='models in experiment')
    parser.add_argument('--net_attack', type=float, default=[1, -1, 0], nargs='+',help='1:attack -1:protect')
    parser.add_argument('--inplace', default='adv_retrain',help='gray model mode: shadow, advretrain,similiar')
    parser.add_argument('--is_gray_net', type=int, default=[1, 0, 0], nargs='+',
                        help='if model i is gray model, the value is 1')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--eps', type=float, default=1 / 8)
    parser.add_argument('--alpha', type=float, default=1 / 20)
    parser.add_argument('--dataset_dir', default='./dataset')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--gpu_id', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = common.get_device()

    assert len(args.nets) == len(args.net_attack)
    assert len(args.nets) == len(args.is_gray_net)

    test_loader = data.get_orig_dataloader(dataset=args.dataset, batch_size=args.batch_size,
                                           dataset_dir=args.dataset_dir, get_train=False)

    def get_shadow_net(x):
        if args.inplace == 'shadow':
            model = gray_net.get_shadow_net(x, dataset=args.dataset)
        if args.inplace == 'similiar':
            model = gray_net.get_similar_net(x, dataset=args.dataset)
        if args.inplace == 'adv_retrain':
            model = gray_net.get_adv_retrain_net(x, dataset=args.dataset)
        return model

    nets = [(models.get_model(x, dataset=args.dataset, is_pretrained=True) if not args.is_gray_net[i]
             else get_shadow_net(x),
             float(args.net_attack[i]), args.is_gray_net[i])
            for i, x in enumerate(args.nets) if args.net_attack[i] != 0]

    orig_nets = [(models.get_model(n, dataset=args.dataset, is_pretrained=True) if args.inplace!='similiar'
                    else models.get_model(n.split('_')[0], dataset=args.dataset, is_pretrained=True))
                    for i, n in enumerate(args.nets)]

    loss_fn = nn.CrossEntropyLoss()
    succ = [0] * len(orig_nets)

    try:
        with tqdm(test_loader, ncols=64) as tq:
            for _, (imgs, labels, fns) in enumerate(tq):
                x, y = imgs.to(device), labels.to(device)
                x_var = Variable(x, requires_grad=True)
                for _ in range(args.max_iter):
                    zero_gradients(x_var)
                    loss = sum([loss_fn(net(x_var), y) * net_attack
                                for i, (net, net_attack, is_shadow) in enumerate(nets)])
                    loss.backward()
                    x_grad = args.alpha * torch.sign(x_var.grad.data)
                    noise = x_var.data + x_grad - x
                    noise = torch.clamp(noise, -args.eps, args.eps)
                    x_adv = torch.clamp(noise + x, -2.2, 2.2)
                    x_var.data = x_adv
                for j, net in enumerate(orig_nets):
                    _, pred = net(x_var).data.max(1)
                    succ[j] += (pred != y).sum().item()
    except KeyboardInterrupt:
        tq.close()
        raise
    tq.close()
    succ = [s / test_loader.dataset.__len__() for s in succ]
    for i, n in enumerate(args.nets):
        print('{} {} succ:{}'.format(n, args.net_attack[i], succ[i]))
