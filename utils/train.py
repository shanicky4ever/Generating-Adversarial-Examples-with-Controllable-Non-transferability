from tqdm import tqdm
import torch
from torch import nn
import visdom
import numpy as np


def test_acc(net, testloader, device='cuda',show_distribute=None,is_half=False):
    acc = 0
    #distribute=[0]*show_distribute if show_distribute else None
    try:
        with tqdm(testloader, ncols=64) as tq:
            for i, (imgs, labels, fn) in enumerate(tq):
                x, y = imgs.to(device) if not is_half else imgs.half().to(device), labels.to(device)
                outputs = net(x)
                _, pred = outputs.max(1)
                acc += pred.eq(y).sum().item()
                '''
                if show_distribute:
                    for j,p in enumerate(np.array(pred.eq(y).cpu())):
                        if not p:
                            distribute[labels[j]]+=1
                '''
    except KeyboardInterrupt:
        tq.close()
        raise
    tq.close()
    print(acc, len(testloader.dataset))
    if show_distribute:
        print(distribute)
    return acc / len(testloader.dataset)


def train(net, trainloader, testloader, epochs=100, lr=1e-2, weight_decay=1e-4, sch_step=None, device='cuda',
          pth_path=None, viz_name=None, train_class=7,show_distribute=None):
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)
    if sch_step:
        sch = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=sch_step,gamma=0.1)
    for e in range(epochs):
        print('batch {} / {}'.format(e + 1, epochs))
        train_loss, correct = 0, 0
        try:
            with tqdm(trainloader, ncols=64) as tq:
                for batch_index, (imgs, labels, fn) in enumerate(tq):
                    x, y = imgs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = net(x)
                    loss = loss_fn(outputs, y)
                    loss.backward()
                    optimizer.step()
                    if sch_step:
                        sch.step()
                    train_loss += loss.item()
                    _, pred = outputs.max(1)
                    correct += pred.eq(y).sum().item()
        except KeyboardInterrupt:
            tq.close()
            raise
        tq.close()
        acc = correct / len(trainloader.dataset)
        losses = train_loss / len(trainloader)
        show = []
        if testloader:
            t_acc = test_acc(net, testloader, device,show_distribute=show_distribute)
            print('correct:{0} train_loss:{1},test_acc:{2}'.format(acc, losses, t_acc))
        else:
            print('correct:{0} train_loss:{1}'.format(acc, losses))

    state = net.state_dict(),
    if pth_path:
        torch.save(state, pth_path)
    return acc