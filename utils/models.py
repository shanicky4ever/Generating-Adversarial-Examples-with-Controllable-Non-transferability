import torch


def get_model(model_name, dataset='cifar10', is_pretrained=True, device='cuda', is_half=False, **kwargs):
    if model_name.endswith('bn'):
        model_name = model_name.replace('bn', '_bn')
    if dataset == 'imagenet':
        return imagenet_model(model_name, is_pretrained, device=device, is_half=is_half, **kwargs)
    else:
        if dataset == 'cifar10':
            from models.cifar10_models import resnet,vgg,densenet
        elif dataset == 'mnist':
            from models.mnist_models import resnet,vgg,densenet
        model_family = None
        for mf in ('resnet','vgg','densenet'):
            if model_name.startswith(mf):
                model_family = mf
        assert model_family is not None
        get_model_commond = '{}.{}(pretrained={}'.format(model_family,model_name, is_pretrained)
        if kwargs:
            for k, v in kwargs.items():
                get_model_commond += ', {}={}'.format(k, v)
        get_model_commond += ')'
        model = eval(get_model_commond).to(device)
        if is_half:
            model = model.half()
        if is_pretrained:
            model.eval()
        return model


def imagenet_model(model_name, is_pretrained, is_half=False, device='cuda', **kwargs):
    from torchvision import models
    get_model_commond = 'models.{}(pretrained={}'.format(model_name, is_pretrained)
    if kwargs:
        for k, v in kwargs.items():
            get_model_commond += ', {}={}'.format(k, v)
    get_model_commond += ')'
    model = eval(get_model_commond).to(device)
    if is_half:
        model = model.half()
    if is_pretrained:
        model.eval()
    return model
