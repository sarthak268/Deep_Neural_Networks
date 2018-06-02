import shutil
from tensorboard_logger import configure, log_value
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
from torchvision.datasets import MNIST
from torch.autograd.gradcheck import zero_gradients
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import math

batchsize = 1

test_set = torch.utils.data.DataLoader(datasets.MNIST('./data',train=False,download=True,transform=transforms.ToTensor()),batch_size=batchsize, shuffle=True)
count = 0

for batch_idx, data in enumerate(test_set):
	img, labels = data
	if ((str)(labels[0])==("tensor(0)")):
		img = img.view(28,28)
		img = img.numpy()
		plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
		plt.savefig('./0/' + str(count) + '.jpg')
	if ((str)(labels[0])==("tensor(1)")):
		img = img.view(28,28)
		img = img.numpy()
		plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
		plt.savefig('./1/' + str(count) + '.jpg')
	if ((str)(labels[0])==("tensor(2)")):
		img = img.view(28,28)
		img = img.numpy()
		plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
		plt.savefig('./2/' + str(count) + '.jpg')
	if ((str)(labels[0])==("tensor(3)")):
		img = img.view(28,28)
		img = img.numpy()
		plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
		plt.savefig('./3/' + str(count) + '.jpg')
	if ((str)(labels[0])==("tensor(4)")):
		img = img.view(28,28)
		img = img.numpy()
		plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
		plt.savefig('./4/' + str(count) + '.jpg')
	if ((str)(labels[0])==("tensor(5)")):
		img = img.view(28,28)
		img = img.numpy()
		plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
		plt.savefig('./5/' + str(count) + '.jpg')
	if ((str)(labels[0])==("tensor(6)")):
		img = img.view(28,28)
		img = img.numpy()
		plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
		plt.savefig('./6/' + str(count) + '.jpg')
	if ((str)(labels[0])==("tensor(7)")):
		img = img.view(28,28)
		img = img.numpy()
		plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
		plt.savefig('./7/' + str(count) + '.jpg')
	if ((str)(labels[0])==("tensor(8)")):
		img = img.view(28,28)
		img = img.numpy()
		plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
		plt.savefig('./8/' + str(count) + '.jpg')
	if ((str)(labels[0])==("tensor(9)")):
		img = img.view(28,28)
		img = img.numpy()
		plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
		plt.savefig('./9/' + str(count) + '.jpg')
	count = count + 1
	print(count)