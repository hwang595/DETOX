"""
Since we need to initialize the dataset in parallel, pre-download data is necessary
This script will do the job for you
"""
import torch
from torchvision import datasets, transforms
from torchvision.datasets import SVHN

if __name__ == "__main__":
	training_set_mnist = datasets.MNIST('./mnist_data', train=True, download=True,
	           transform=transforms.Compose([
	               transforms.ToTensor(),
	               transforms.Normalize((0.1307,), (0.3081,))]))
	train_loader_mnist = torch.utils.data.DataLoader(training_set_mnist, batch_size=128, shuffle=True)
	test_loader_mnist = torch.utils.data.DataLoader(
	    datasets.MNIST('./mnist_data', train=False, transform=transforms.Compose([
	               transforms.ToTensor(),
	               transforms.Normalize((0.1307,), (0.3081,))
	           ])), batch_size=100, shuffle=True)
	trainset_cifar10 = datasets.CIFAR10(root='./cifar10_data', train=True,
	                                        download=True, transform=transforms.Compose([
	               transforms.ToTensor(),
	               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	           ]))
	train_loader_cifar10 = torch.utils.data.DataLoader(trainset_cifar10, batch_size=128,
	                                          shuffle=True)
	test_loader_cifar10 = torch.utils.data.DataLoader(
	    datasets.CIFAR10('./cifar10_data', train=False, transform=transforms.Compose([
	               transforms.ToTensor(),  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	           ])), batch_size=100, shuffle=True)

	training_set = SVHN('./svhn_data', split='train', download=True, transform=transforms.Compose([
	                         transforms.RandomCrop(32, padding=4),
	                         transforms.RandomHorizontalFlip(),
	                         transforms.ToTensor(),
	                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	                     ]))
	train_loader = torch.utils.data.DataLoader(training_set, batch_size=128,
	                                          shuffle=True)
	transform_test = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	    ])
	testset = SVHN(root='./svhn_data', split='test',
	                                       download=True, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=1000,
	                                         shuffle=False)
	# Cifar-100 dataset
	normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
	                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
	transform_train = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Lambda(lambda x: F.pad(
	                        Variable(x.unsqueeze(0), requires_grad=False, volatile=True),
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
	train_loader = torch.utils.data.DataLoader(training_set, batch_size=128,
	                                          shuffle=True)
	testset = datasets.CIFAR100(root='./cifar100_data', train=False,
	                                       download=True, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=1000,
	                                         shuffle=False)