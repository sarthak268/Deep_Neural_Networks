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
from random import randint

batchsize = 75

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

model = VAE(784,450,200,20)
model = load_model()

def find_jacobian(model, z1): #Jh
	z = z1
	dec = Variable(model.decode(z).data, requires_grad=True)
	enc1, enc2 = model.encode(dec)
	jacobian = torch.FloatTensor(20,784).zero_()
	for j in range(20):
		f = torch.FloatTensor(20).zero_()
		f[j] = 1	
		enc1.backward(f, retain_graph=True)
		jacobian[j,:] = dec.grad.data
		dec.grad.data.zero_()
	return jacobian

def find_jacobian_1(model, z1): #Jg
	z = z1
	dec = model.decode(z)
	jacobian = torch.FloatTensor(784,20).zero_()
	for j in range(784):
		f = torch.FloatTensor(784).zero_()
		f[j] = 1	
		dec.backward(f, retain_graph=True)
		jacobian[j,:] = z.grad.data
		z.grad.data.zero_()
	return jacobian

def sample(model):
	array1 = []
	array2 = []
	folder = path.realpath("./0")
    images = os.listdir(folder)
	count = 0
	for i in images:
		img = img.view(img.size(0), -1)
		img = Variable(img)
		z,z_ = model.encode(img)
		z1 = z[k,:]
		z1 = Variable(z1, requires_grad=True)
		j1 = find_jacobian(model,z1)
		u1, sigma1, vh1 = torch.svd(j1)
		array1.append(sigma1)
		j2 = find_jacobian_1(model,z1)
		u2, sigma2, vh2 = torch.svd(j2)
		array2.append(sigma2)
		count += 1
		if(count==75):
			break
	data1 = []
	data2 = []

	for j in range(20):
		sum1 = 0
		sum2 = 0
		for i in range(0,len(array1)):
			pl = array1[i].numpy()
			sum1 += pl[j]
		sum1 = sum1/100
		data1.append(sum1)

		for i in range(0,len(array2)):
			pl1 = array2[i].numpy()
			sum2 += pl1[j]
		sum2 = sum2/100
		data2.append(sum2)
	return data1, data2

d1,d2 = sample(model)
print("d1",d1)
print("d2",d2)

plt.plot(d1)
plt.show()
plt.savefig("jacobian.jpg")


# plt.plot(d2)
# plt.show()
# plt.savefig("jacobian_1.jpg")
















