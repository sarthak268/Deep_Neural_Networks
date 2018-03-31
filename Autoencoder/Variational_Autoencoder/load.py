import torch
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

class VAE(nn.Module):
    def __init__(self,n1,n2,latent_dimension):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(n1, n2)
        self.fc21 = nn.Linear(n2, latent_dimension)
        self.fc22 = nn.Linear(n2, latent_dimension)
        self.fc3 = nn.Linear(latent_dimension, n2)
        self.fc4 = nn.Linear(n2, n1)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
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
model.load_state_dict(torch.load('./vae.pth'))

