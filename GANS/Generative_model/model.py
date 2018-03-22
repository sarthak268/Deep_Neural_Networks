import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0., 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1., 0.02)
        m.bias.data.fill_(0.)

class DCGenerator(nn.Module):
    def __init__(self, convs):
        super(DCGenerator, self).__init__()
        self.convs = nn.ModuleList()
        in_channels = 1
        for i, (out_channels, kernel_size, stride, padding) in enumerate(convs):
            self.convs.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
            if i < len(convs)-1:
                # we use BN and RELU for each layer except the output
                self.convs.append(nn.BatchNorm2d(out_channels))
                self.convs.append(nn.ReLU())
            else:
                # in output, we use Tanh to generate data in [-1, 1]
                self.convs.append(nn.Tanh())
            in_channels = out_channels
        self.apply(init_weight)

    def forward(self, input):
        out = input
        for module in self.convs:
            out = module(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, convs):
        super(Discriminator, self).__init__()
        self.convs = nn.ModuleList()
        in_channels = 1
        for i, (out_channels, kernel_size, stride, padding) in enumerate(convs):
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
            if i != 0 and i != len(convs)-1:
                # we donot use BN in the input layer of D
                self.convs.append(nn.BatchNorm2d(out_channels))
            if i != len(convs)-1:
                self.convs.append(nn.LeakyReLU(0.2))
                in_channels = out_channels
        #self.cls = nn.Linear(out_channels*in_width*in_height, nout)
        self.apply(init_weight)

    def forward(self, input):
        out = input

        for layer in self.convs:
            out = layer(out)
        out = out.view(out.size(0), -1)
        out = F.sigmoid(out)
        return out