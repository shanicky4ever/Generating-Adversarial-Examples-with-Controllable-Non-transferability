# Generating Adversarial Examples with Controllable Non-transferability

Here is the code of paper -- Generating Adversarial Examples with Controllable Non-transferability

## Prepare

This experiment code bases on Python3.7, the package we need is

    torch torchvision tqdm h5py tqdm

You can run

    pip install -r requirements.txt

to install the packages, or you can install them by conda.

## Dataset

In this code, we can run with Cifar10 and Imagenet. Cifar10 need not do anything more, the dataset will be downloaded automatically. Because of torchvision no longer provide Imagenet api directly, we use `ImageFolder` instead. If you want to run with Imagenet, please make a dir `dataset/imagenet` and download ILSVRC2012 dataset from [Imagenet website](http://www.image-net.org/challenges/LSVRC/2012/downloads) in it, extract the tar file and run [valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) to preprocess.

## Run

We use argprase to control the parameters in experiments, for examples, you can change default dataset `cifar10` to `Imagenet` by

    python xxx.py --dataset imagenet


For all code, you can run 

    python xxx.py -h

for parameters details.

Here we give basic commands for the experiments.

## Reversed Loss Function Ensemble

### White-box attack

To attack white-box attack, suppose attack `resnet50` and protect `vgg16bn`, while monitor the transferability to `densenet121`, you can run

    python RLFE.py --nets resnet50 vgg16bn densenet121 --net_attack 1 -1 0 --is_gray_net 0 0 0

### Defense model 

To attack Defense model while protect normal model, for example, `resnet50` and `resnet50 defense`. Firstly, train a defense model.

    python defense_model.py --mode get_adv --attack_model resnet50
    python defense_model.py --mode adv_retrain --attack_model resnet50

And then, run with

    python RLFE.py --inplace adv_retrain --nets resnet50 resnet50 --net_attack 1 -1 --is_gray_net 0 1

### Shadow model

To attack shadow model, train a shadow model firstly

    python train_net.py --model resnet50

Suppose protect `resnet50` by `resnet50 shadow`, attack `vgg16bn`, and monitor `densenet121`

    python RLFE.py --inplace shadow --nets resnet50 vgg16bn densenet121 --net_attack -1 1 0 --is_gray_net 1 0 0

### Similiar family

Suppose protect `resnet50` by `resnet18`, attack `vgg16bn`, and monitor `densenet121`

    python RLFE.py --inplace similiar --nets resnet50_18 vgg16bn densenet121 --net_attack -1 1 0 --is_gray_net 1 0 0

## Transferability Classification black-box attack

Suppose attack `resnet50` as white-box model, to protect black-box `vgg16bn` while keep the transferability to black-box `densenet121`. This method rely on the origin category, here we only show the code of category 0.

Firstly, attack with `resenet50`,

    python TC --mode get_advs

Then, train the classifier,

    python TC.py --mode noise_classifier_train --select_category 0

Finally, do the directly attack. Here, since the differences between different classes are small, and there are some random factors in the operation, we only give a best possible parameter selection in our experiment

    python TC.py --mode direct_attack --select_category 0 --eps 0.01 --alpha 0.001 --max_iter 20

