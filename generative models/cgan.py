import torch
import torch.nn as nn
from torch.nn import functional as F

from args import get_parser
# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

nz = opts.nz
nc = opts.nc
ndf = opts.ndf
ngf = opts.ngf

class Generator(nn.Module):
    # initializers
    def __init__(self):
        super(Generator, self).__init__()
        # z, state size: [100, 1, 1]
        self.deconv1_1 = nn.ConvTranspose2d(nz, ngf*4, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(ngf*4)
        # state size: [ngf*4, 4, 4]

        # z label, state size: [10, 1, 1]
        self.deconv1_2 = nn.ConvTranspose2d(10, ngf*4, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(ngf*4)
        # state size: [ngf*4, 4, 4]

        # after concat: state size: [ngf*8, 4, 4]
        self.deconv2 = nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1) 
        self.deconv2_bn = nn.BatchNorm2d(ngf*4)
        # state size: [ngf*4, 8, 8]
        self.deconv3 = nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(ngf*2)
        # state size: [ngf*2, 16, 16]
        self.deconv4 = nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(ngf)
        # state size: [ngf, 32, 32]
        self.deconv5 = nn.ConvTranspose2d(ngf, 1, 4, 2, 1)
        # state size: [1, 64, 64]

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.deconv1_1_bn(self.deconv1_1(input)), 0.2) # => 
        y = F.leaky_relu(self.deconv1_2_bn(self.deconv1_2(label)), 0.2) # => 
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
        x = torch.tanh(self.deconv5(x))
        return x


class Discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(Discriminator, self).__init__()
        # image, state size: [1, 64, 64]
        self.conv1_1 = nn.Conv2d(1, ndf//2, 4, 2, 1)
        # state size: [ndf/2, 32, 32]

        # image label, state size: [10, 64, 64]
        self.conv1_2 = nn.Conv2d(10, ndf//2, 4, 2, 1)
        # state size: [ndf/2, 32, 32]

        # after concat, state size: [ndf, 32, 32]
        self.conv2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(ndf*2)
        # state size: [ndf*2, 16, 16]
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(ndf*4)
        # state size: [ndf*4, 8, 8]
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(ndf*8)
        # state size: [ndf*8, 4, 4]
        self.conv5 = nn.Conv2d(ndf*8, 1, 4, 1, 0)
        # state size: [ndf*8, 1, 1]

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        return x
