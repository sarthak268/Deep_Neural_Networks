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
from random import randint
from os import path
import matplotlib.image as mpimg


batchsize = 100

class VAE(nn.Module):
	def __init__(self,n1,n2,n3,latent_dimension):
		super(VAE, self).__init__()

		self.fc1 = nn.Linear(n1, n2)
		self.fc11 = nn.Linear(n2,n3)
		self.fc21 = nn.Linear(n3, latent_dimension)
		self.fc22 = nn.Linear(n3, latent_dimension)
		self.fc3 = nn.Linear(latent_dimension, n3)
		self.fc33 = nn.Linear(n3,n2)
		self.fc4 = nn.Linear(n2, n1)

	def encode(self, x):
		# h
		h11 = F.relu(self.fc1(x))
		h1 = F.relu(self.fc11(h11))
		return self.fc21(h1), self.fc22(h1)

	def reparametrize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps)
		return eps.mul(std).add_(mu)

	def decode(self, z):
		# g
		h33 = F.elu(self.fc3(z)) #20-200
		h3 = F.elu(self.fc33(h33)) #200-450
		return F.sigmoid(self.fc4(h3)) #450-784

	def get_latent_variable(self, mu, logvar):
		z = self.reparametrize(mu, logvar)
		return z

	def forward(self, x):
		global mean
		global log_variance
		mu, logvar = self.encode(x)
		mean = mu
		log_variance = logvar
		z = self.reparametrize(mu, logvar)
		return self.decode(z), mu, logvar


def load_model():
	model.load_state_dict(torch.load('./vae.pth'))
	return model

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


model = VAE(784,450,200,20)
model = load_model()

classwise_folder = '../unlabelled/plots/data/'

for j in range(10):
	number = str(j)
	folder = path.realpath(classwise_folder + 'resized_' + number)
	images = os.listdir(folder)
	arr = torch.zeros(20,len(images))
	count = 0
	for i in images:
		img = mpimg.imread(classwise_folder + 'resized_' + number + '/' + i)
		img = rgb2gray(img)
		img = img.reshape(784)
		img = Variable(torch.FloatTensor(img))
		z, _ = model.encode(img)
		arr[:,count] = z
		count += 1
	mean = arr.mean(dim=1)
	print ('mean for ' + number + ' is ' + str(mean))
	std = arr.std(dim=1)
	print ('std for ' + number + ' is ' + str(std))
	
	
	

	






























