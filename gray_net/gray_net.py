import os
import torch
from utils import models


def get_shadow_net(model_name, dataset='cifar10',device='cuda', **kwargs):
    model = models.get_model(model_name, dataset=dataset,is_pretrained=False, device=device)
    script_dir = os.path.dirname(__file__)
    model.load_state_dict(torch.load(os.path.join(script_dir, 'state_dicts', '_'.join(
            ['my',model_name,dataset])+'.pt'), map_location=device)[0])
    model = model.to(device)
    model.eval()
    return model


def get_similar_net(model_name,dataset='cifar10', device='cuda', **kwargs):
    new_model ={'vgg16bn_11':'vgg11bn','vgg16bn_19':'vgg19bn',
                'resnet50_18':'resnet18','resnet50_34':'resnet34'}
    model = models.get_model(new_model[model_name], dataset=dataset,is_pretrained=True, device=device)
    return model

def get_adv_retrain_net(model_name,dataset='cifar10',device='cuda',**kwargs):
    model = models.get_model(model_name,dataset=dataset,is_pretrained=False,device=device)
    script_dir = os.path.dirname(__file__)
    model.load_state_dict(torch.load(os.path.join(script_dir, 'state_dicts', '_'.join(
        [model_name,dataset, 'defense'])+'.pt'), map_location=device)[0])
    model = model.to(device)
    model.eval()
    return model