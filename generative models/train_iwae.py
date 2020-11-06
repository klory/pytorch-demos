from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import make_grid

from utils import clean_state_dict

from iwae import IWAE_1
from args import get_parser
from torch.distributions.binomial import Binomial
import numpy as np
from tqdm import tqdm
# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

if opts.manualSeed is None:
    opts.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opts.manualSeed)
random.seed(opts.manualSeed)
torch.manual_seed(opts.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opts.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


dataset = dset.MNIST(root=opts.dataroot, download=True, train=True,
                    transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batchSize,
                                         shuffle=False, num_workers=int(opts.workers),
                                         pin_memory=True)
dataset = dset.MNIST(root=opts.dataroot, download=True, train=False,
                    transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batchSize,
                                         shuffle=False, num_workers=int(opts.workers),
                                         pin_memory=True)

device = torch.device("cuda" if opts.cuda else "cpu")

model = IWAE_1(opts.nz, 784).to(device)
model.double()
if torch.cuda.device_count() > 1 and opts.cuda:
    print("Let's use {} GPUs".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model_name = 'iwae_{}samples_{}seed_{}nz'.format(
        len(train_loader.dataset), 
        opts.manualSeed, 
        opts.nz)
model_path = os.path.join('checkpoints', model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

from tensorboardX import SummaryWriter
writer = SummaryWriter(os.path.join('runs', model_name))
n_iter = 0
num_samples = 50
sample = torch.randn(64, opts.nz).double().to(device)
for epoch in range(opts.epochs):
    print('=> Epoch {}'.format(epoch))
    model.train()
    running_loss = []
    for data in tqdm(train_loader):
        image = data[0]
        m = Binomial(1, image.view(-1, 784))
        # inputs = m.sample(torch.Size([num_samples])).double().to(device)
        inputs = m.sample().expand(num_samples, image.shape[0], 784).double().to(device)
        optimizer.zero_grad()
        loss, bce, kld = model.train_loss(inputs)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        writer.add_scalar('bce', bce, n_iter)
        writer.add_scalar('kld', kld, n_iter)
        writer.add_scalar('loss', loss, n_iter)
        n_iter += 1

    writer.add_scalar('loss_epoch', np.mean(running_loss), epoch)
    
    model.eval()
    running_loss = []
    for data in tqdm(test_loader):
        image = data[0]
        m = Binomial(1, image.view(-1, 784))
        # inputs = m.sample(torch.Size([num_samples])).double().to(device)
        inputs = m.sample().expand(500, image.shape[0], 784).double().to(device)
        loss = model.test_loss(inputs)  
        running_loss.append(loss.item())
    writer.add_scalar('nll_epoch', np.mean(running_loss), epoch)
    
    image = model.decoder(sample).cpu()
    image = make_grid(image.view(64,1,28,28), nrow=8, normalize=True, scale_each=True)
    writer.add_image('random images', image, epoch)

    if (epoch + 1)%10 == 0 or epoch+1 == opts.epochs:
        torch.save(model.state_dict(),
                   os.path.join(model_path, 'e{}.pth').format(epoch))