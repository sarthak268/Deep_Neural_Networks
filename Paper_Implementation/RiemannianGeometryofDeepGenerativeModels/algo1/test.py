import numpy as np

import theano # comment / uncomment
import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

model = nn.Sequential(
    nn.Linear(784, 1000),
    nn.ReLU(),
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 10))

x = Variable(torch.rand(100, 784), requires_grad=True)
y = model(x)

grad_var = torch.zeros(*y.size())
grad_var[:, 0] = 1
y.backward(grad_var, retain_variables=True)



