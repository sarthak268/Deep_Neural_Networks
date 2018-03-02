import torch 
from torch import autograd, nn
import torch.nn.functional as f

batchsize = 964
# total images
inputsize = 32 * 32
# number of features of pixels in this case
outputsize = 10
# total number of classes (like number of different images - 0-9 for digits)
hiddendim = 100

Input = autograd.Variable(torch.randn(batchsize, inputsize))
# variable for storing input 
Output = autograd.Variable(torch.randn(batchsize, outputsize), requires_grad = False)
# variable for storing output

print ("Input = " , Input)
print ("Output = " , Output)

model = torch.nn.Sequential(
	torch.nn.Linear(inputsize, hiddendim),
	torch.nn.ReLU(),
	# rectified linear unit that removes all negative values and replaces them with 0
	torch.nn.Linear(hiddendim, outputsize),
	)

lossfunction = torch.nn.MSELoss(size_average = False)
# mean square loss function

learningrate = 1/(10000)

for i in range (500):
	predictedoutput = model(Input)
	# forward pass

	loss = lossfunction(predictedoutput,Output) 
	#print (i,loss.data[0])
	if (i == 0):
		print("Initial loss = ", loss.data[0])

	model.zero_grad()
	# zeros the gradient before running backwards pass

	loss.backward()
	# compute gradient for all learnable parameters in the pass

	# number of paramters for the model are 4
	for param in model.parameters():
		param.data = param.data - learningrate*param.grad.data
	# applying gradient descent

#print ("Model parameters", model.parameters())
print ("Final loss = " , loss.data[0])
