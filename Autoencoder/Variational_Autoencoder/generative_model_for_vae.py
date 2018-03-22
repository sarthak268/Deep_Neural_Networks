import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from glob import glob
#from util import *
import numpy as np
from PIL import Image

# Parsers
parser = argparse.ArgumentParser(description='PyTorch VAE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

torch.manual_seed(args.seed)
args.cuda = False
kwargs = {}

train_loader = range(2080)
test_laoder = range(40)

toTensor = transforms.ToTensor()	# transforms into a tensor

def loadBatch(batch_index, isTrainingSet):
	if (isTrained):
		template = '../data/train/%s.jpg'
	else:
		template = '../data/test/%s.jpg'
	l = [str(batch_index * 128 + i).zfill(6) for i in range(128)]	# zfill : fills zeros before the string till the string length becomes equal to given paramter 
	data = []

	for index in l:
		img = Image.open(template%index)
		data.append(np.array(img))
	data = [toTensor(i) for i in data]
	return torch.stack(data, dim = 0)	# concatenates all tensors along a new tensor

class VAE(nn.Module):
	def __init__(self, nc, ngf, ndf, latent_variable_size):
		super(VAE, self).__init__()

		self.nc = nc
		self.ngf = ngf
		self.ndf = ndf 
		self.latent_variable_size = latent_variable_size

		# Encoder
		self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
		self.bn1 = nn.BatchNorm2d(ndf)
		self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
		self.bn2 = nn.BatchNorm2d(ndf*2)
		self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
		self.bn3 = nn.BatchNorm2d(ndf*4)
		self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
		self.bn4 = nn.BatchNorm2d(ndf*8)
		self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
		self.bn5 = nn.BatchNorm2d(ndf*8)
		self.fc1 = nn.Linear(ndf*8*4*4, latent_variable_size)
		self.fc2 = nn.Linear(ndf*8*4*4, latent_variable_size)

        # Batch normalization means subtracting bacth mean and divinding by bactch standard deviation
        # https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c

        # Decoder
		self.d1 = nn.Linear(latent_variable_size, ngf*8*2*4*4)
		self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
		self.pd1 = nn.ReplicationPad2d(1)
		self.d2 = nn.Conv2d(ngf*8*2, ngf*8, 3, 1)
		self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)
		self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
		self.pd2 = nn.ReplicationPad2d(1)
		self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
		self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)
		self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
		self.pd3 = nn.ReplicationPad2d(1)
		self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
		self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)
		self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
		self.pd4 = nn.ReplicationPad2d(1)
		self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
		self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)
		self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
		self.pd5 = nn.ReplicationPad2d(1)
		self.d6 = nn.Conv2d(ngf, nc, 3, 1)
		self.leakyrelu = nn.LeakyReLU(0.2)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

		def Encode(self, x):
			h1 = self.leakyrelu(self.bn1(self.e1(x)))
			h2 = self.leakyrelu(self.bn2(self.e2(h1)))
			h3 = self.leakyrelu(self.bn3(self.e3(h2)))
			h4 = self.leakyrelu(self.bn4(self.e4(h3)))
			h5 = self.leakyrelu(self.bn5(self.e5(h4)))
			h5 = h5.view(-1, self.ndf*8*4*4)

			return self.fc1(h5), self.fc2(h5)

		def reparameterize(self, mu, logvariance):
			std = logvariance.mul(0.5).exp_()
			eps = torch.FloatTensor(std.size()).normal_()
			eps = Variable(eps)
			return eps.mul(std).add_(mu)

		def Decode(self, z):
			h1 = self.relu(self.d1(z))
			h1 = h1.view(-1, self.ngf*8*2, 4, 4)
			h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
			h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
			h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
			h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

			return self.sigmoid(self.d6(self.pd5(self.up5(h5))))

		def forward(self, x):
			mu, logvariance = self.Encode(x.view(-1, self.nc, self.ndf, self.ngf))
			z = self.reparameterize(mu, logvariance)
			result = self.Decode(z)
			return result, mu, logvariance

		def getLatentVariables(self, x):
			mu, logvariance = self.Encode(x.view(-1, self.nc, self.ndf, self.ngf))
			z = self.reparameterize(mu, logvariance)
			return z


model = VAE(nc = 3, ngf = 128, ndf = 128, latent_variable_size = 500)

reconstructionFunction = nn.BCELoss()	# Binary cross entropy loss
reconstructionFunction.size_average = False

def lossFunction(reconstrctedX, x, mu, logvariance):
	BCE = reconstructionFunction(reconstructedX, x)
	KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
	KLD = torch.sum(KLD_element).mul_(-0.5)
	return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr = 1e-4)

