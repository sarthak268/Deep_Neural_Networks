import torch
from torch import autograd, nn
import torch.nn.functional as f
import argparse
import torchvision.datasets as datasets 
import torchvision.transforms as transforms

root = './data'
download = False

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = datasets.MNIST(root=root, train=True, transform=trans, download=download)
test_set = datasets.MNIST(root=root, train=False, transform=trans)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

#print ('==>>> total trainning batch number: ',  (len(train_loader)))
#print ('==>>> total testing batch number: ',  (len(test_loader)))

class Encoder(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Encoder, self).__init__()

	def forward(self, x):


