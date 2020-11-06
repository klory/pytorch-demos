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
from torchvision.utils import save_image, make_grid

from tensorboardX import SummaryWriter
from vae import VAE
from utils import clean_state_dict, weights_init
from tqdm import tqdm
import numpy as np

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

dataset = torch.utils.data.Subset(dataset, [0,1,2])
train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batchSize,
                                         shuffle=False, num_workers=int(opts.workers),
                                         pin_memory=True)
dataset = dset.MNIST(root=opts.dataroot, download=True, train=False,
                    transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batchSize,
                                         shuffle=False, num_workers=int(opts.workers),
                                         pin_memory=True)

device = torch.device("cuda" if opts.cuda else "cpu")

model = VAE().to(device)
if torch.cuda.device_count() > 1 and opts.cuda:
    print("Let's use {} GPUs".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model_name = 'vae_{}samples_{}seed_{}nz'.format(
        len(train_loader.dataset), 
        opts.manualSeed, 
        opts.nz)
model_path = os.path.join('checkpoints', model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE/x.shape[0], KLD/x.shape[0]

writer = SummaryWriter(os.path.join('runs', model_name))
n_iter = 0

def train(epoch):
    losses = []
    global n_iter
    print('=> Epoch {}'.format(epoch))
    model.train()
    train_loss = 0
    for data in tqdm(train_loader):
        image = data[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(image)
        bce, kld = loss_function(recon_batch, image, mu, logvar)
        loss = bce + kld
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        losses.append(loss.item())
        writer.add_scalar('bce', bce, n_iter)
        writer.add_scalar('kld', kld, n_iter)
        writer.add_scalar('loss', loss, n_iter)

        writer.add_histogram('mu', mu, n_iter)
        writer.add_histogram('std', torch.exp(0.5*logvar), n_iter)
        n_iter += 1

    writer.add_scalar('loss_epoch', np.array(losses).mean(), epoch)


def test(epoch):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            bce, kld = loss_function(recon_batch, data, mu, logvar)
            loss = bce+kld
            losses.append(loss.item())
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(opts.batchSize, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         os.path.join(model_path, 'reconstruction_' + str(epoch) + '.png'), nrow=n)

    writer.add_scalar('nll_epoch', np.array(losses).mean(), epoch)

if __name__ == "__main__":
    from scipy.stats import norm
    grid_x = norm.ppf(np.linspace(0.0000001, 0.9999999, 64))
    sample = torch.tensor(grid_x).float().to(device).view(64,1)
    # sample = torch.randn(64, opts.nz).to(device)
    for epoch in range(0, opts.epochs):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            image = model.decode(sample).cpu()
            image = make_grid(image.view(64,1,28,28), nrow=8, normalize=True, scale_each=True)
            writer.add_image('random images', image, epoch)