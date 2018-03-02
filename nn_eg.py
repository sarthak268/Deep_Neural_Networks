import torch
from torch import autograd, nn
import torch.nn.functional as F

batch_size = 964	# total no of inputs
input_size = 32 * 32	# number of input features
hidden_size = 100	# features for hidden layer
num_classes = 10	# number of classes
 
torch.manual_seed(123)
input = autograd.Variable(torch.rand(batch_size, input_size))
target_data = autograd.Variable((torch.rand(batch_size) * num_classes).long())
#print ("target",target_data)

#print ('input', input)

class Net(nn.Module):
	 def __init__(self, input_size, hidden_size, num_classes):
	 	super().__init__()
	 	self.h1 = nn.Linear(input_size, hidden_size)
	 	self.h2 = nn.Linear(hidden_size, num_classes)

	 def forward(self, x): 
	 	x = self.h1(x)
	 	x = F.tanh(x)
	 	x = self.h2(x)
	 	x = F.softmax(x)
	 	return x 


model = Net(input_size= input_size, hidden_size=hidden_size, num_classes=num_classes)
out = model(input)
#print ('output', out)
_, pred = out.max(1)
#print (pred)