import argparse
import os
from utils import attack, common, data, train, models
from utils import compare_data as cdata
import torch
from tqdm import tqdm
import pickle
import numpy as np
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='cifar10')
    #parser.add_argument('--mode', default='get_advs')
    #parser.add_argument('--mode', default='noise_classifier_train')
    parser.add_argument('--mode', default='direct_attack')
    parser.add_argument('--is_train',default='True')
    parser.add_argument('--white_model', default='resnet50')
    parser.add_argument('--black_model', default=['vgg16bn', 'densenet121'], nargs='+')
    parser.add_argument('--classifier_model', default='resnet18')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_iter', type=int, default=3)
    parser.add_argument('--eps', type=float, default=1 / 8)
    parser.add_argument('--alpha', type=float, default=1 / 20)
    parser.add_argument('--dataset_dir', default='./dataset')
    parser.add_argument('--root_folder', default='exp')
    parser.add_argument('--adv_folder', default='normal_advs')
    parser.add_argument('--divid_folder', default='divid')
    parser.add_argument('--select_category', type=int, default=0)
    parser.add_argument('--quad', default=[1, 2, 3, 4], nargs='+')
    parser.add_argument('--quad_multi', type=int, default=[1, 5, 1, 5], nargs='+')
    parser.add_argument('--target_quad', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--classifier_pth_folder', default='pth')
    parser.add_argument('--save_pth',default='True')
    parser.add_argument('--tmp_noise_folder', default='tmp_noise')
    parser.add_argument('--tmp_divid_folder', default='tmp_divid')
    parser.add_argument('--save_every_step',default='False')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = common.get_device()

    root_folder = args.root_folder + '_' + args.dataset
    data.make_dir(root_folder, del_before=False)
    adv_folder = os.path.join(root_folder, args.adv_folder)
    divid_folder = os.path.join(root_folder, args.divid_folder)
    this_white_attack_folder = os.path.join(adv_folder, args.white_model)
    classifer_pth_folder = os.path.join(root_folder, args.classifier_pth_folder)
    tmp_noise_folder = os.path.join(root_folder, args.tmp_noise_folder)
    tmp_divid_folder = os.path.join(root_folder, args.tmp_divid_folder)
    tmp_adv_folder = os.path.join(root_folder, 'tmp_adv')

    save_every_step = args.save_every_step=='True'
    save_pth = args.save_pth=='True'
    is_train = args.is_train=='True'

    if args.mode == 'get_advs':
        data.make_dir(adv_folder, del_before=False)
        trainloader, testloader = data.get_orig_dataloader(batch_size=args.batch_size, dataset=args.dataset,
                                                           dataset_dir=args.dataset_dir,get_train=True)
        white_model = models.get_model(dataset=args.dataset, model_name=args.white_model,is_pretrained=True)
        black_model = [models.get_model(b, is_pretrained=True,dataset=args.dataset) for b in args.black_model]
        for dataset in ('test','train'):
            this_white_attack_folder_set=this_white_attack_folder+'_{}'.format(dataset)
            data.make_dir(this_white_attack_folder_set, del_before=True)
            dataloader = trainloader if dataset=='train' else testloader
            trainset_succ = attack.i_fgsm(dataloader=dataloader, model=white_model, model_name=args.white_model,
                                    max_iter=args.max_iter, eps=args.eps, alpha=args.alpha,
                                    device=device, output_dir=this_white_attack_folder_set, save_every_step=save_every_step,
                                    dataset=args.dataset)
            res_loader = data.get_folder_dataloader(dataset_folder=this_white_attack_folder_set, batch_size=args.batch_size,
                                                    root=args.dataset_dir,dataset=args.dataset)
            distribute = [0] * len(args.quad)
            black_res = {}
            for net_id, bn in enumerate(black_model):
                try:
                    with tqdm(res_loader, ncols=64) as tq:
                        for i, (img, label, fns) in enumerate(tq):
                            output = bn(img.to(device))
                            _, pred = output.max(1)
                            res = np.array(pred.eq(label.to(device)).cpu())
                            for j, fn in enumerate(fns):
                                if not net_id:
                                    black_res[fn] = {b: None for b in args.black_model}
                                black_res[fn][args.black_model[net_id]] = not res[j]
                except KeyboardInterrupt:
                    tq.close()
                    raise
                tq.close()
            for fn, r in black_res.items():
                [b1, b2] = args.black_model
                if r[b1] and r[b2]:
                    distribute[0]+=1
                elif not (r[b1] or r[b2]):
                    distribute[2] += 1
                else:
                    distribute[1 if not r[b1] else 3] += 1
            print(distribute)
            with open(os.path.join(this_white_attack_folder_set, 'succ.pkl'), 'wb') as f:
                pickle.dump(black_res, f, protocol=pickle.HIGHEST_PROTOCOL)

    if args.mode == 'noise_classifier_train':
        data.make_dir(classifer_pth_folder, del_before=False)
        num_classes = len(args.quad)
        print(this_white_attack_folder)
        trainloader, testloader = cdata.get_quad_loader(dataset_folder=this_white_attack_folder+'_train',
                                                        batch_size=args.batch_size, select_category=args.select_category,
                                                        white_model=args.white_model, black_model=args.black_model,
                                                        divid=True, quad=args.quad, quad_multi=tuple(args.quad_multi),
                                                        dataset=args.dataset)
        classifier = models.get_model(model_name=args.classifier_model, is_pretrained=False, device=device,
                                      num_classes=num_classes,dataset=args.dataset)
        pth_name = '_'.join(
            [args.classifier_model, args.white_model] + args.black_model + [str(args.select_category)]) + '.pt'
        train.train(net=classifier, trainloader=trainloader, testloader=testloader, lr=args.lr,
                    weight_decay=args.weight_decay, epochs=args.epochs,
                    pth_path=os.path.join(classifer_pth_folder, pth_name) if save_pth else None,
                    train_class=args.select_category)
    if args.mode == 'direct_attack':
        data.make_dir(tmp_noise_folder, del_before=True)
        data.make_dir(tmp_divid_folder, del_before=True)
        data.make_dir(tmp_adv_folder,del_before=True)

        classifier = models.get_model(args.classifier_model, is_pretrained=False, device=device,
                                      num_classes=len(args.quad),dataset=args.dataset)
        pth_name = '_'.join([args.classifier_model, args.white_model] + args.black_model + [str(args.select_category)]) + '.pt'
        state_dict = torch.load(os.path.join(classifer_pth_folder, pth_name), map_location=device)[0]
        classifier.load_state_dict(state_dict)
        classifier.eval()
        black_model = [models.get_model( b, is_pretrained=True, device=device, dataset=args.dataset)
                                        for b in args.black_model]
        white_model = models.get_model(args.white_model, is_pretrained=True, device=device, dataset=args.dataset)
        
        noise_loader = cdata.get_noise_divid_loader(dataset_folder=this_white_attack_folder+'_test',        
                                                    classifier=classifier,
                                                    batch_size=args.batch_size, select_category=args.select_category,
                                                    device=device, tmp_folder=tmp_divid_folder,
                                                    dataset=args.dataset, black_model=black_model)
        target = args.quad.index(args.target_quad)
        
        attack_succ = attack.i_fgsm_target( dataloader=noise_loader, net=classifier, 
                                            net_name=args.classifier_model,
                                            target=target, num_classes=len(args.quad),
                                            max_iter=args.max_iter, eps=args.eps, alpha=args.alpha, 
                                            device=device,
                                            output_dir=tmp_noise_folder, save_every_step=False,
                                            dataset=args.dataset)
        noise2adv_loader = cdata.noise_to_adv_loader(dataset_folder=tmp_noise_folder, batch_size=args.batch_size,
                                                     dataset=args.dataset)
        
        succ=0
        try:
            with tqdm(noise2adv_loader, ncols=64) as tq:
                for i, (imgs, labels, fns) in enumerate(tq):
                    x = imgs.to(device)
                    y= labels.to(device)
                    output = black_model[0 if args.target_quad==2 else 1](x)
                    _, p = output.data.max(1)
                    succ += (p != y).sum()
                    data.save_attack_img(imgs=imgs,dataset=args.dataset,fn=fns,attack_method='re',
                                         output_dir=tmp_adv_folder)
        except KeyboardInterrupt:
            tq.close()
            raise
        tq.close()
        print('protect model transfer down to {}'.format(succ.cpu().numpy()/noise2adv_loader.dataset.__len__()))
