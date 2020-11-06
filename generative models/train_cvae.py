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
from torchvision.utils import save_image

from tensorboardX import SummaryWriter
from cvae import CVAE
from utils import clean_state_dict, weights_init

from args import get_parser
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

model = CVAE().to(device)
if torch.cuda.device_count() > 1 and opts.cuda:
    print("Let's use {} GPUs".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model_name = 'cvae_{}samples_{}seed_{}nz'.format(
        len(train_loader.dataset), 
        opts.manualSeed, 
        opts.nz)
model_path = os.path.join('checkpoints', model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, yh):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - (mu-yh).pow(2) - logvar.exp())
    return BCE + KLD

from tqdm import tqdm
def train(epoch):
    model.train()
    train_loss = 0
    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, yh = model(data, label.long())
        loss = loss_function(recon_batch, data, mu, logvar, yh)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            recon_batch, mu, logvar, yh = model(data, label.long())
            test_loss += loss_function(recon_batch, data, mu, logvar, yh).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(opts.batchSize, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         os.path.join(model_path, 'reconstruction_' + str(epoch) + '.png'), nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, opts.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            # fixed noise & label
            temp_z_ = torch.randn(10, opts.nz)
            fixed_z_ = temp_z_
            fixed_y_ = torch.zeros(10, 1)
            for i in range(9):
                fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
                temp = torch.ones(10, 1) + i
                fixed_y_ = torch.cat([fixed_y_, temp], 0)
            
            z_y = model.fcy(fixed_y_.to(device).long()).squeeze() # [100, nz]
            fixed_z_ = z_y + fixed_z_.to(device)
            sample = model.decode(fixed_z_).cpu()
            save_image(sample.view(100, 1, 28, 28),
                       os.path.join(model_path, 'sample_' + str(epoch) + '.png'), nrow=10)