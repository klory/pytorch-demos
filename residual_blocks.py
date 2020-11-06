import torch
from torch import nn

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

def conv_bn_act(in_channels, out_channels, stride=1, act=nn.ReLU()):
    return nn.Sequential(
        conv3x3(in_channels, out_channels, stride),
        nn.BatchNorm2d(out_channels), 
        act
    )

# Residual block
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if (stride != 1) or (in_channels != out_channels):
            self.downsample = nn.Sequential(
                conv3x3(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# deconvolution
def deconv3x3(in_channels, out_channels, stride=1):
    if stride == 1:
        return nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    elif stride == 2:
        return nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=stride, padding=1)

def deconv_bn_act(in_channels, out_channels, stride=1, act=nn.ReLU()):
    return nn.Sequential(
        deconv3x3(in_channels, out_channels, stride),
        nn.BatchNorm2d(out_channels), 
        act
    )

# Residual block up
class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockUp, self).__init__()
        self.deconv1 = deconv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dconv2 = deconv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.upsample = None
        if (stride != 1) or (in_channels != out_channels):
            self.upsample = nn.Sequential(
                deconv3x3(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        
    def forward(self, x):
        residual = x
        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dconv2(out)
        out = self.bn2(out)
        if self.upsample:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)
        return out


if __name__ == '__main__':
    device = 'cuda'
    bs = 4

    x = torch.randn(bs, 3, 256, 256).to(device)
    residuel_block = ResidualBlock(3, 16, stride=2).to(device)
    out = residuel_block(x)
    print(out.shape)

    x = torch.randn(bs, 16, 128, 128).to(device)
    residuel_block_up = ResidualBlockUp(16, 3, stride=2).to(device)
    out = residuel_block_up(x)
    print(out.shape)
