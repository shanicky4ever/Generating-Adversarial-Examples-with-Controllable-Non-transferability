import torch

def get_device(gpu_number=0):
    if torch.cuda.is_available():
        print('Use {}'.format(torch.cuda.get_device_name(gpu_number)))
        return 'cuda'
    else:
        print('no gpu used')
        return 'cpu'