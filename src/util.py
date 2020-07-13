import random

import numpy as np
from torchvision import datasets, transforms

from model_ops.lenet import LeNet, LeNetSplit
from model_ops.resnet import *
from model_ops.resnet_split import *
from model_ops.vgg import *
from model_ops.densenet import *
from model_ops.fc_nn import FC_NN, FC_NN_Split
from model_ops.utils import err_simulation

from torchvision.datasets import SVHN
from coding import search_w
from master import baseline_master, rep_master, cyclic_master, draco_lite_master
from worker import baseline_worker, rep_worker, cyclic_worker, draco_lite_worker

SEED_ = 428

def build_model(model_name, num_classes):
    # build network
    if model_name == "LeNet":
        return LeNet()
    elif model_name == "ResNet18":
        return ResNet18()
    elif model_name == "ResNet34":
        return ResNet34()
    elif model_name == "ResNet50":
        return ResNet50()
    elif model_name == "VGG11":
        return vgg11_bn(num_classes=num_classes)
    elif model_name == "VGG13":
        return vgg13_bn(num_classes=num_classes)
    elif model_name == "DenseNet":
        return DenseNet121()



def load_data(dataset, seed, args):
    if seed:
        # in normal method we do not implement random seed here
        # same group should share the same shuffling result
        torch.manual_seed(seed)
        random.seed(seed)
    if dataset == "MNIST":
        training_set = datasets.MNIST('./mnist_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))]))
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
    elif dataset == "Cifar10":
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        # data prep for training set
        # note that the key point to reach convergence performance reported in this paper (https://arxiv.org/abs/1512.03385)
        # is to implement data augmentation
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                                Variable(x.unsqueeze(0), requires_grad=False),
                                (4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        # load training and test set here:
        training_set = datasets.CIFAR10(root='./cifar10_data', train=True,
                                                download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                                  shuffle=True)
        testset = datasets.CIFAR10(root='./cifar10_data', train=False,
                                               download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                                 shuffle=False)
    elif args.dataset == 'SVHN':
        training_set = SVHN('./svhn_data', split='train', transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ]))
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                                  shuffle=True)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        testset = SVHN(root='./svhn_data', split='test',
                                               download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                                 shuffle=False)
    elif args.dataset == "Cifar100":
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                                Variable(x.unsqueeze(0), requires_grad=False),
                                (4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        # load training and test set here:
        training_set = datasets.CIFAR100(root='./cifar100_data', train=True,
                                                download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                                  shuffle=True)
        testset = datasets.CIFAR100(root='./cifar100_data', train=False,
                                               download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                                 shuffle=False)
    return train_loader, training_set, test_loader


def group_assign(world_size, group_size, rank):
    if world_size % group_size == 0:
        ret_group_dict, group_list = _assign(world_size, group_size, rank)  
    else:
        ret_group_dict, group_list = _assign(world_size-1, group_size, rank)
        group_list[-1].append(world_size)
    group_num, group_seeds = _group_identify(group_list, rank)
    return ret_group_dict, group_num, group_seeds
    

def _group_identify(group_list, rank):
    group_seeds = [0]*len(group_list)
    if rank == 0:
        return -1, group_seeds
    for i,group in enumerate(group_list):
        group_seeds[i] = np.random.randint(0, 20000)
        if rank in group:
            group_num = i
    return group_num, group_seeds


def _assign(world_size, group_size, rank):
    np.random.seed(SEED_)
    ret_group_dict={}
    k = int(world_size/group_size)
    group_list=[[j+i*group_size+1 for j in range(group_size)] for i in range(k)]
    for i, l in enumerate(group_list):
        ret_group_dict[i]=l
    return ret_group_dict, group_list


def _generate_adversarial_nodes(args, world_size):
    # generate indices of adversarial compute nodes randomly at each iteration
    np.random.seed(SEED_)
    return [np.random.choice(np.arange(1, world_size), size=args.worker_fail, replace=False) for _ in range(args.max_steps+1)]


def prepare(args, rank, world_size):
    device = torch.device("cpu")
    if args.approach == "baseline":
        # randomly select adversarial nodes
        adversaries = _generate_adversarial_nodes(args, world_size)
        train_loader, training_set, test_loader = load_data(dataset=args.dataset, seed=None, args=args)
        kwargs_master = {
                    'batch_size':args.batch_size, 
                    'learning_rate':args.lr, 
                    'max_epochs':args.epochs, 
                    'max_steps':args.max_steps, 
                    'momentum':args.momentum, 
                    'network':args.network,
                    'comm_method':args.comm_type, 
                    'worker_fail':args.worker_fail,
                    'eval_freq':args.eval_freq, 
                    'train_dir':args.train_dir, 
                    'update_mode':args.mode, 
                    'compress_grad':args.compress_grad, 
                    'checkpoint_step':args.checkpoint_step,
                    'lis_simulation':args.lis_simulation,
                    'device':device
                    }
        kwargs_worker = {
                    'update_mode':args.mode,  # for implementing signSGD
                    'batch_size':args.batch_size, 
                    'learning_rate':args.lr, 
                    'max_epochs':args.epochs, 
                    'max_steps':args.max_steps,
                    'momentum':args.momentum, 
                    'network':args.network,
                    'comm_method':args.comm_type, 
                    'adversery':args.adversarial, 
                    'worker_fail':args.worker_fail,
                    'err_mode':args.err_mode, 
                    'compress_grad':args.compress_grad, 
                    'eval_freq':args.eval_freq, 
                    'train_dir':args.train_dir, 
                    'checkpoint_step':args.checkpoint_step,
                    'lis_simulation':args.lis_simulation,
                    'adversaries':adversaries,
                    'device':device
                    }
    # majority vote
    elif args.approach == "maj_vote":
        adversaries = _generate_adversarial_nodes(args, world_size)
        group_list, group_num, group_seeds=group_assign(world_size-1, args.group_size, rank)
        train_loader, training_set, test_loader = load_data(dataset=args.dataset, seed=group_seeds[group_num], args=args)
        kwargs_master = {
                    'batch_size':args.batch_size, 
                    'learning_rate':args.lr, 
                    'max_epochs':args.epochs, 
                    'max_steps':args.max_steps, 
                    'momentum':args.momentum, 
                    'network':args.network,
                    'comm_method':args.comm_type, 
                    'eval_freq':args.eval_freq, 
                    'train_dir':args.train_dir, 
                    'group_list':group_list, 
                    'update_mode':args.mode, 
                    'compress_grad':args.compress_grad, 
                    'checkpoint_step':args.checkpoint_step,
                    'device':device
                    }
        kwargs_worker = {
                    'batch_size':args.batch_size, 
                    'learning_rate':args.lr, 
                    'max_epochs':args.epochs, 
                    'max_steps':args.max_steps,
                    'momentum':args.momentum, 
                    'network':args.network,
                    'comm_method':args.comm_type, 
                    'adversery':args.adversarial, 
                    'worker_fail':args.worker_fail,
                    'err_mode':args.err_mode, 
                    'group_list':group_list, 
                    'group_seeds':group_seeds, 
                    'group_num':group_num,
                    'compress_grad':args.compress_grad, 
                    'eval_freq':args.eval_freq, 
                    'train_dir':args.train_dir,
                    'adversaries':adversaries,
                    'device':device
                    }
    # draco lite
    elif args.approach == "draco_lite":
        adversaries = _generate_adversarial_nodes(args, world_size)
        group_list, group_num, group_seeds=group_assign(world_size-1, args.group_size, rank)
        train_loader, training_set, test_loader = load_data(dataset=args.dataset, seed=group_seeds[group_num], args=args)
        kwargs_master = {
                    'batch_size':args.batch_size, 
                    'learning_rate':args.lr, 
                    'max_epochs':args.epochs, 
                    'max_steps':args.max_steps, 
                    'momentum':args.momentum, 
                    'network':args.network,
                    'comm_method':args.comm_type, 
                    'eval_freq':args.eval_freq, 
                    'train_dir':args.train_dir, 
                    'group_list':group_list, 
                    'update_mode':args.mode,
                    'bucket_size':args.bucket_size,
                    'compress_grad':args.compress_grad, 
                    'checkpoint_step':args.checkpoint_step,
                    'lis_simulation':args.lis_simulation,
                    'worker_fail':args.worker_fail,
                    'device':device
                    }
        kwargs_worker = {
                    'update_mode':args.mode,  # for implementing signSGD
                    'batch_size':args.batch_size, 
                    'learning_rate':args.lr, 
                    'max_epochs':args.epochs, 
                    'max_steps':args.max_steps,
                    'momentum':args.momentum, 
                    'network':args.network,
                    'comm_method':args.comm_type, 
                    'adversery':args.adversarial, 
                    'worker_fail':args.worker_fail,
                    'err_mode':args.err_mode, 
                    'group_list':group_list, 
                    'group_seeds':group_seeds, 
                    'group_num':group_num,
                    'compress_grad':args.compress_grad, 
                    'eval_freq':args.eval_freq, 
                    'train_dir':args.train_dir,
                    'adversaries':adversaries,
                    'lis_simulation':args.lis_simulation,
                    'device':device
                    }
    datum = (train_loader, training_set, test_loader)
    return datum, kwargs_master, kwargs_worker