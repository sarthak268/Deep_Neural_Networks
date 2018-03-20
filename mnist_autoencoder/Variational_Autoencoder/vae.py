import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
from torchvision.datasets import MNIST
import os
import random

array = []

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 50
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#dataset = MNIST('./data', transform=img_transform, download=True)
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def write(arr,file):
    for i in range (len(arr)):
        for j in range (3):
            file.write(str(arr[i][j]) + " ")
        file.write("\n")    


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()

reconstruction_function = nn.MSELoss(size_average=False)

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(batchsize):
    train_set = torch.utils.data.DataLoader(datasets.MNIST('./data',train=True,download=True,transform=transforms.ToTensor()),batch_size=batchsize, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_set):
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img)
        
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(img)
            loss = loss_function(recon_batch, img, mu, logvar)
            loss.backward()
            train_loss += loss.data[0]
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(img),
                    len(train_set.dataset), 
                    100. * batch_idx / len(train_set),
                    loss.data[0] / len(img)))

            ########################################
            array.append([epoch, loss.data[0] / len(img), 100. * batch_idx / len(train_set)])
                # epoch, loss, percentage

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_set.dataset)))
        if epoch % 10 == 0:
            save = to_img(recon_batch.cpu().data)
            save_image(save, './vae_img/image_{}.png'.format(epoch))

    file = open("data_vae.txt","w") 
    write(array,file)
    file.close()
    return model

'''file = open("data_vae.txt","w") 
write(array,file)
file.close()
'''
#return model

def test(model,batchsize):
    test_set=torch.utils.data.DataLoader(datasets.MNIST('./data',train=False,download=True,transform=transforms.ToTensor()),batch_size=batchsize,shuffle=False)
    r=random.randint(0,len(test_set.dataset))
    test_image=test_set.dataset[r][0]
    input_image=test_image.view(28,28).numpy()
    out, mu1, logvar1 = model(Variable(test_image))
    output_image = out.view(28,28).data.numpy()

    plot_image = np.concatenate((input_image, output_image), axis=1) 
    plt.imshow(plot_image, cmap='gray', interpolation='nearest');
    plt.show()
    

model = train(batchsize = batch_size)
#test(model,batchsize = batch_size)
torch.save(model.state_dict(), './vae.pth')















































