import random 
import torch 
from torch.autograd import Variable 

class DynamicNet(torch.nn.Module):
	def __init__(self, inputdimension, hiddendimension, outputdimension):
		super(Net, self).__init__()
		self.linear1 = torch.nn.Linear(inputdimension, hiddendimension)
		self.linear2 = torch.nn.Linear(hiddendimension, hiddendimension)
		self.linear3 = torch.nn.Linear(hiddendimension, outputdimension)

	def forward(self, x):
		relu = self.linear1(x).clamp(min = 0)
		for i in range (random.randint(0,3)):
			relu = self.linear2(relu).clamp( min = 0)
			y_predicted = self.linear3(relu)
			return y_predicted

bacthsize = 64
inputdimension = 1000
hiddendimension = 100
outputdimension = 10

x = Variable(torch.randn(bacthsize, inputdimension))
y = Variable(torch.randn(bacthsize, outputdimension), requires_grad = False)

model = DynamicNet(inputdimension, hiddendimension, outputdimension)

criteron = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4, momentum = 0.9)

for j in range (500):
	y_pred = model(x)
	loss = criteron(y_pred, y)
	print (t, loss.data[0])

	optimizer.zero_grad()
	loss.backwards()
	optimizer.step()