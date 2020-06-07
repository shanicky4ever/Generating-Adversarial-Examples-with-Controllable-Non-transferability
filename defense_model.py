import argparse
from utils import data, common, attack, models, train
import os
from tqdm import tqdm
import pickle
from utils import compare_data as cdata
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--mode', default='get_adv')
    parser.add_argument('--mode', default='adv_retrain')
    parser.add_argument('--attack_model', default='resnet50')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_iter', type=int, default=5)
    parser.add_argument('--eps', type=float, default=1 / 16)
    parser.add_argument('--alpha', type=float, default=1 / 40)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--path', default='tmp_defense')
    parser.add_argument('--gpu_number', default='0')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--pth_path',default='gray_net/state_dicts')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_number
    device = common.get_device()
    data.make_dir(args.path, del_before=False)
    path = os.path.join(args.path, args.dataset)
    data.make_dir(path, del_before=False)
    path = os.path.join(path, args.attack_model)
    data.make_dir(path, del_before=False)

    if args.mode == 'get_adv':
        adv_imgs_path = os.path.join(path, 'adv_imgs')
        data.make_dir(adv_imgs_path, del_before=True)
        train_loader, test_loader = data.get_orig_dataloader(batch_size=args.batch_size, 
                                dataset=args.dataset, is_tran=True,get_train=True)
        model = models.get_model(args.attack_model, dataset=args.dataset, is_pretrained=True, device=device)
        acc = attack.i_fgsm(dataloader=train_loader, model=model, model_name=args.attack_model,
                             max_iter=args.max_iter, eps=args.eps, alpha=args.alpha, 
                             output_dir=adv_imgs_path, save_every_step=False, dataset=args.dataset)
        print(acc)
    if args.mode == 'adv_retrain':
        model = models.get_model(model_name=args.attack_model, dataset=args.dataset, is_pretrained=False, device=device)
        adv_imgs_path = os.path.join(path, 'adv_imgs')
        data.make_dir(args.pth_path, del_before=False)
        adv_retrain_pth_path = os.path.join(args.pth_path, 
                            '_'.join([args.attack_model, args.dataset,'defense']) + '.pt')
        folder_loader,test_loader = data.multiple_dataloader(dataset_folder=adv_imgs_path, dataset=args.dataset,
                                                 batch_size=args.batch_size, divid=True, is_tran=True)
        sch_step = [50, 120]
        acc = train.train(  model, trainloader=folder_loader, testloader=test_loader, 
                            epochs=args.epochs, lr=args.lr,
                            weight_decay=args.weight_decay, sch_step=sch_step, device=device,
                            pth_path=adv_retrain_pth_path)
        print(acc)
