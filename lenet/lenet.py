from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np

from sklearn.utils import shuffle
import pandas as pd

class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, (5,5), padding=2)
		self.conv2 = nn.Conv2d(6, 16, (5,5))
		self.fc1   = nn.Linear(16*5*5, 120)
		self.fc2   = nn.Linear(120, 84)
		self.fc3   = nn.Linear(84, 10)
	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

net = LeNet()
print (net)

# use_gpu = torch.cuda.is_available()
# if use_gpu:
# 	net = net.cuda()
# 	print ('USE GPU')
# else:
# 	print ('USE CPU')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

print ("1. Loading data")
train = pd.read_csv("/Users/sarthakbhagat/Desktop/math_task/codes/lenet/train.csv").values
train = shuffle(train)
test  = pd.read_csv("/Users/sarthakbhagat/Desktop/math_task/codes/lenet/test.csv").values

print ("2. Converting data")
X_data  = train[:, 1:].reshape(train.shape[0], 1, 28, 28)
X_data  = X_data.astype(float)
X_data /= 255.0
X_data  = torch.from_numpy(X_data);
X_label = train[:,0];
X_label = X_label.astype(int);
X_label = torch.from_numpy(X_label);
X_label = X_label.view(train.shape[0],-1);
print (X_data.size(), X_label.size())

print ("3. Training phase")
nb_train = train.shape[0]
nb_epoch = 75000
nb_index = 0
nb_batch = 4

for epoch in range(nb_epoch):
	if nb_index + nb_batch >= nb_train:
		nb_index = 0
	else:
		nb_index = nb_index + nb_batch

	mini_data  = Variable(X_data[nb_index:(nb_index+nb_batch)].clone())
	mini_label = Variable(X_label[nb_index:(nb_index+nb_batch)].clone(), requires_grad = False)
	mini_data  = mini_data.type(torch.FloatTensor)
	mini_label = mini_label.type(torch.LongTensor)
	# if use_gpu:
	# 	mini_data  = mini_data.cuda()
	# 	mini_label = mini_label.cuda()
	optimizer.zero_grad()
	mini_out   = net(mini_data)
	mini_label = mini_label.view(nb_batch)
	mini_loss  = criterion(mini_out, mini_label)
	mini_loss.backward()
	optimizer.step() 

	if (epoch + 1) % 2000 == 0:
		print("Epoch = %d, Loss = %f" %(epoch+1, mini_loss.data[0]))

print ("4. Testing phase")

Y_data  = test.reshape(test.shape[0], 1, 28, 28)
Y_data  = Y_data.astype(float)
Y_data /= 255.0
Y_data  = torch.from_numpy(Y_data);
print (Y_data.size())
nb_test = test.shape[0]

net.eval()

final_prediction = np.ndarray(shape = (nb_test, 2), dtype=int)
for each_sample in range(nb_test):
	sample_data = Variable(Y_data[each_sample:each_sample+1].clone())
	sample_data = sample_data.type(torch.FloatTensor)
	# if use_gpu:
	# 	sample_data = sample_data.cuda()
	sample_out = net(sample_data)
	_, pred = torch.max(sample_out, 1)
	final_prediction[each_sample][0] = 1 + each_sample
	final_prediction[each_sample][1] = pred.data[0]
	if (each_sample + 1) % 2000 == 0:
		print("Total tested = %d" %(each_sample + 1))

print ('5. Generating submission file')

submission = pd.DataFrame(final_prediction, dtype=int, columns=['ImageId', 'Label'])
submission.to_csv('/Users/sarthakbhagat/Desktop/math_task/codes/lenet/output.csv', index=False, header=True)
