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

num_epochs = 1
batch_size = 128
learning_rate = 1e-3

mean = Variable(torch.zeros(128,20))
log_variance = Variable(torch.zeros(128,20))

img_transform = transforms.Compose([transforms.ToTensor()])

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

class VAE(nn.Module):
    def __init__(self,n1,n2,latent_dimension):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(n1, n2)
        self.fc21 = nn.Linear(n2, latent_dimension)
        self.fc22 = nn.Linear(n2, latent_dimension)
        self.fc3 = nn.Linear(latent_dimension, n2)
        self.fc4 = nn.Linear(n2, n1)

    def encode(self, x):
        # h
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        # g
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

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

model = VAE(784,400,20)

reconstruction_function = nn.MSELoss(size_average=False)

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(batchsize):
    train_set = torch.utils.data.DataLoader(datasets.MNIST('./data',train=True,download=True,transform=transforms.ToTensor()),batch_size=batchsize, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_set):
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(img)
            loss = loss_function(recon_batch, img, mu, logvar)
            loss.backward()
            train_loss += loss.data[0]
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(img),
                    len(train_set.dataset), 
                    100. * batch_idx / len(train_set),
                    loss.data[0] / len(img)))

            ########################################
            #array.append([epoch, loss.data[0] / len(img), 100. * batch_idx / len(train_set)])
                # epoch, loss, percentage
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_set.dataset)))
        if epoch % 10 == 0:
            save = to_img(recon_batch.cpu().data)
            save_image(save, './vae_img/image_{}.png'.format(epoch))
    return model

def load_model():
	model.load_state_dict(torch.load('./vae.pth'))
	return model

def save_model(model):
	torch.save(model.state_dict(), './vae.pth')

def linear_interpolation(z0, zt):
    # z0 and zt in FloatTensor
    z0n = z0.detach().numpy()
    ztn = zt.detach().numpy()
    z_middle = np.zeros(z0n.shape)
    for i in range(z0n.shape[0]):
        z_middle[i] = random.uniform(min(z0n[i], ztn[i]), max(z0n[i], ztn[i]))
    z_middle_t = torch.from_numpy(z_middle)
    return z_middle_t.float()

def find_jacobian(model, z1):#, zi):
	#z = Variable(torch.FloatTensor([[2,1]]), requires_grad=True)
	#z = Variable(torch.FloatTensor(20).normal_(), requires_grad=True)
	#z = Variable(z1, requires_grad=True)
	z = z1
	dec = model.decode(z)
	enc1, enc2 = model.encode(dec)
	#print(enc)
	jacobian = torch.FloatTensor(20,784).zero_()
	for j in range(2):
		f = torch.FloatTensor(20).zero_()
		f[j] = 1
		for i in range(7):	
			enc1.backward(f, retain_graph=True)
			jacobian[:,i] = z.grad.data
			z.grad.data.zero_()
		print("yoyo")
	return jacobian


T = 10
epsilon = 0.1
z_collection = []
delta_e = torch.FloatTensor(20,784).zero_()

def find_etta_i(model,z0, z1, z2):
    dt = 1 / T
    e = -(1 / dt)*find_jacobian(model,z1)*((model.decode(z2) - 2*model.decode(z1)+model.decode(z0)).data)
    #e = find_jacobian(model,z1)*(model.decode(z2).data)
    #print(e)
    return e

def sum_energy(model):
	delta_e = torch.FloatTensor(20,784).zero_()
	for i in range(1,T-2):
		delta_e = find_etta_i(model,z_collection[i-1],z_collection[i],z_collection[i+1])
	return(torch.mm(math.abs(delta_e),math.abs(delta_e)))

def main(model,z0,zt):
    step_size = 0.001
    z_collection.append(z0)
    
    for i in range(T-2):
        w = (linear_interpolation(z0,zt))
        z_collection.append(w)
    z_collection.append(zt)

    #while (sum_energy(model) > epsilon):
    #	for i in range(1,T):
    #    	etta_i = find_etta_i(z_collection[i-1], z_collection[i], z_collection[i+1])
        	#print("jooooooooooooo")
    #         z_collection[i] = z_collection[i] - step_size*etta_i
    return z_collection
        


#############################################################################
# TRAINING A NEW MODEL
#train(batchsize = batch_size)
#save_model(model)
#############################################################################

#############################################################################
# LOADING EXISTING MODEL
load_model()
#############################################################################


z0 = Variable(torch.FloatTensor(20).normal_(), requires_grad=True)
zt = Variable(torch.FloatTensor(20).normal_(), requires_grad=True)
zt1 = Variable(torch.FloatTensor(20).normal_(), requires_grad=True)
#find_etta_i(model=model,z0=z0,z1=zt,z2=zt1)
main(model=model,z0=z0, zt=zt)
#find_etta_i(model,z_collection[0],z_collection[1],z_collection[2])
find_etta_i(model=model,z0=z0,z1=zt,z2=zt1)
#sum_energy(model=model)
#find_jacobian(model=model, z1=zt)




















