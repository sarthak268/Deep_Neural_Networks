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

#==========================================================

class Net(nn.Module):
    def __init__(self, batch_size, input_size):
            super(Net, self).__init__()
            self.l1 = nn.Linear(input_size, 100)
            self.l2 = nn.Linear(100, input_size)
    def forward(self, x):
            encoded = self.l1(x)
            decoded = self.l2(encoded)
            return encoded, decoded

'''class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.UpsamplingNearest2d(scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

class Lenet(nn.Module):
    def __init__(self, batch_size, input_size):
        super(Lenet, self).__init__()
        self.conv1 = ConvLayer(1, 6, kernel_size=5, stride=1)
        self.conv2 = ConvLayer(6, 10, kernel_size=5, stride=1)
        self.l1 = nn.Linear(10*5*5, 120)
        self.l2 = nn.Linear(120, 60)
        self.l3 = nn.Linear(60, 10)
        
        self.l4 = nn.Linear(10, 60)
        self.l5 = nn.Linear(60, 120)
        self.l6 = nn.Linear(120, 10*5*5)
        self.deconv1 = UpsampleConvLayer(6, 10, kernel_size=5, stride=1, upsample=2)
        self.deconv2 = UpsampleConvLayer(1, 6, kernel_size=5, stride=1, upsample=2)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = x.view(-1, 10*5*5)
        x = f.relu(self.l1(x))
        x = f.relu(self.l2(x))
        encoded = self.l3(x)

        y = encoded
        y = f.relu(self.l4(y))
        y = f.relu(self.l5(y))
        y = f.relu(self.l6(y))
        y = y.view(10,5*5)
        y = f.relu(self.deconv1(y))
        decoded = self.deconv2(y)

        return encoded ,decoded
'''

net = Net(batch_size=batch_size,input_size=input_size)

optimizer = torch.optim.Adam(net.parameters(),lr=10**(-4))
loss_function = nn.MSELoss()

# hyperparameters
EPOCH = 50
N_TEST_IMG = 5

f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()

view_data = Variable(train_set.train_data[:N_TEST_IMG].view(-1,28*28).type(torch.FloatTensor)/255.)

#=====================================================================================================
# For linear one layered nn
'''
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
                        f.savefig('/Users/sarthakbhagat/Desktop/Neural_Nets/mnist_autoencoder/images/'+ str(epoch) +'.jpg')
                plt.pause(0.05)
'''
#======================================================================================================









































