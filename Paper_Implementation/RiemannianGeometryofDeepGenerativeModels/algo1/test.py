import numpy as np

#import theano # comment / uncomment
import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

x = Variable(torch.FloatTensor([[2,1]]), requires_grad=True)
M = Variable(torch.FloatTensor([[1,2],[3,4]]))
y = x*M
#print(y)
# jacobian = torch.FloatTensor(2, 2).zero_()
# y.backward(torch.FloatTensor([[1, 0]]), retain_graph=True)
# jacobian[:,0] = x.grad.data
# print(jacobian)
# x.grad.data.zero_()
# y.backward(torch.FloatTensor([[0, 1]]), retain_graph=True)
# jacobian[:,1] = x.grad.data
# print(jacobian)

#z = Variable(torch.FloatTensor(20).normal_(), requires_grad=True)
#dec = model.decode(z)
#enc1, enc2 = model.encode(dec)
#print(enc)
jacobian = torch.FloatTensor(2,2).zero_()
for j in range(2):
	f = torch.FloatTensor(2).zero_()
	f[j] = 1
	#print(f)
	for i in range(2):	
		y.backward(f, retain_graph=True)
		jacobian[:,i] = x.grad.data
		x.grad.data.zero_()
print(jacobian)