import torch
from torch import autograd, nn
import torch.nn.functional as f
import argparse
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable 
import numpy as np
import matplotlib
matplotlib.use('Agg')

root = './data'
download = False

#trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
trans = transforms.ToTensor()
train_set = datasets.MNIST(root=root, train=True, transform=trans, download=download)
test_set = datasets.MNIST(root=root, train=False, transform=trans)

batch_size = 64
input_size = 28*28

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

#==========================================================

#input data set image plot

#plt.imshow(train_set.train_data[100].numpy(), cmap='gray')
#plt.show()

#==========================================================

#print ('==>>> total trainning batch number: ',  (len(train_loader))) # 600
#print ('==>>> total testing batch number: ',  (len(test_loader))) # 100

# print (train_set.train_data.size())   # 60,000
# print (test_set.test_data.size())     # 10,000 

# images are 28*28

class Net(nn.Module):
        def __init__(self, batch_size, input_size):
                super(Net, self).__init__()
                self.l1 = nn.Linear(input_size, 100)
                self.l2 = nn.Linear(100, input_size)
        def forward(self, x):
                encoded = self.l1(x)
                decoded = self.l2(encoded)
                return encoded, decoded

net = Net(batch_size=batch_size,input_size=input_size)

optimizer = torch.optim.Adam(net.parameters(),lr=10**(-4))
loss_function = nn.MSELoss()

# hyperparameters
EPOCH = 50
N_TEST_IMG = 5

f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()

view_data = Variable(train_set.train_data[:N_TEST_IMG].view(-1,28*28).type(torch.FloatTensor)/255.)

for i in range (N_TEST_IMG):
        a[0][i].imshow(np.reshape(view_data.data.numpy()[i],(28,28)),cmap = "gray")
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())

for epoch in range(EPOCH):
        count = 0
        for step, (x,y) in enumerate(train_loader):
                batch_x = Variable(x.view(-1, 28*28))
                batch_y = Variable(x.view(-1, 28*28))
                batch_label = Variable(y)

                encoded, decoded = net(batch_x)

                loss = loss_function(decoded, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step % 100 == 0):
                        print ("epoch = ", epoch, " train loss = ", loss.data[0])
                        count = count + 1

                _, decoded_data = net(view_data)
                for i in range(N_TEST_IMG):
                        a[1][i].clear()
                        a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28,28)), cmap='gray')
                        a[1][i].set_xticks(())
                        a[1][i].set_yticks(())
                plt.draw()
                if (count == 1):
                        f.savefig('/Users/sarthakbhagat/Desktop/Neural_Nets/mnist_autoencoder/images'+ str(epoch) +'.jpg')
                plt.pause(0.05)









































