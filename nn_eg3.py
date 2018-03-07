import torch 
from torch import autograd, nn
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets 

class NN(nn.Module):
	def __init__(self,input_size,hidden_size,output_size):
		super(NN,self).__init__()