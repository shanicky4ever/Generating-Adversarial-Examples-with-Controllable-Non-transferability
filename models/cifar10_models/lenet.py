'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import os
import torch

class LeNet(nn.Module):
    def __init__(self,num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def lenet(pretrained,device,**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = LeNet(**kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(script_dir + '/state_dicts/'+'alex.pt', map_location=device)
        model.load_state_dict(state_dict)
    return model