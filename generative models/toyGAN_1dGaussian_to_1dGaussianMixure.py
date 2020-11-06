import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pdb
import os

import argparse
import sklearn

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--nh', default=64, type=int)
parser.add_argument('--bs', default=64, type=int)
args = parser.parse_args()

save_dir = 'outputs/seed{}_nh{}_bs{}'.format(args.seed, args.nh, args.bs)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
print('seed = {}'.format(seed))

# https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
nh = args.nh
G = nn.Sequential(
    nn.Linear(1,nh),
    nn.ReLU(True),
    nn.Linear(nh,nh),
    nn.ReLU(True),
    nn.Linear(nh,nh),
    nn.ReLU(True),
    nn.Linear(nh,1),
)

D = nn.Sequential(
    nn.Linear(1,nh),
    nn.ReLU(True),
    nn.Linear(nh,nh),
    nn.ReLU(True),
    nn.Linear(nh,nh),
    nn.ReLU(True),
    nn.Linear(nh,1),
    nn.Sigmoid()
)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

G.apply(weights_init)
D.apply(weights_init)

loss_fn = nn.BCELoss()
optimizerG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=[0, 0.9])
optimizerD = torch.optim.Adam(D.parameters(), lr=1e-4, betas=[0, 0.9])
batch_size = args.bs

epochs = 10000

def sample_GM_1d(ws, mus, stds, num_samples):
    """
    generate 1d Gaussian Mixture samples
    argements:
        ws -- weights of Bernoulli, tensor with shape [N]
        mus -- means of each Gaussian, tensor with shape [N]
        stds -- stds of each Gaussian, tensor with shape [N]
        num_samples -- number of samples, scalar
    return:
        tensor of size [number_samples]
    """
    assert ws.shape[0] == mus.shape[0]
    assert mus.shape[0] == stds.shape[0]
    assert mus.dtype == torch.float32
    assert ws.dtype == torch.float32
    m = torch.distributions.Multinomial(1, ws)
    batch = []
    for _ in range(num_samples):
        idx = m.sample().nonzero()[0]
        mu = mus[idx]
        std = stds[idx]
        value = torch.randn(1)*std + mu
        batch.append(value)
    # change to tensor
    batch = torch.tensor(batch).view(-1,1)
    return batch


ws_z = torch.tensor([1.0])
mus_z = torch.tensor([0.0])
stds_z = torch.tensor([1.0])

# ws_x = torch.ones(4)
# mus_x = torch.tensor([-4,-1,1,4]).float()*2
# stds_x = torch.ones(4)*0.01

# ws_x = torch.ones(1).float()
# mus_x = torch.tensor([4]).float()
# stds_x = torch.ones(1).float() * 0.3

ws_x = torch.ones(2).float()
mus_x = torch.tensor([-4, 4]).float()
stds_x = torch.ones(2).float() * 0.5

# fixed z
z_fix = sample_GM_1d(ws_z, mus_z, stds_z, 1000)
# fixed real
real_fix = sample_GM_1d(ws_x, mus_x, stds_x, 1000)

# draw G and D output
grid_z = np.linspace(-3, 3, 1000)
grid_z = torch.tensor(grid_z).float().view(1000,1)

tmp, tmp_idx = torch.min(mus_x, 0)
left = tmp-3*stds_x[tmp_idx]
tmp, tmp_idx = torch.max(mus_x, 0)
right = tmp+3*stds_x[tmp_idx]
grid_real = np.linspace(left.item(), right.item(), 1000)
grid_real = torch.tensor(grid_real).float().view(1000,1)

D_reals = []
D_fakes_1 = []
D_fakes_2 = []

device = torch.device('cuda')
G = G.to(device)
D = D.to(device)

plt.figure(figsize=(16,8))
for epoch in tqdm(range(epochs)):
    real = sample_GM_1d(ws_x, mus_x, stds_x, batch_size).to(device)
    z = sample_GM_1d(ws_z, mus_z, stds_z, batch_size).to(device)
    label_real = torch.ones(batch_size, 1).to(device)
    label_fake = torch.zeros(batch_size, 1).to(device)
    # optimize D multiple times
    for _ in range(5):
        optimizerD.zero_grad()
        D_real = D(real)
        D_fake_1 = D(G(z))
        loss_D = loss_fn(D_real, label_real) + loss_fn(D_fake_1, label_fake)
        loss_D.backward()
        optimizerD.step()

    z = sample_GM_1d(ws_z, mus_z, stds_z, batch_size).to(device)
    # optimize G multiple times
    for _ in range(1):
        optimizerG.zero_grad()
        fake = G(z)
        D_fake_2 = D(fake)
        loss_G = loss_fn(D_fake_2, label_real)
        loss_G.backward()
        optimizerG.step()

    D_reals.append(D_real.mean().item())
    D_fakes_1.append(D_fake_1.mean().item())
    D_fakes_2.append(D_fake_2.mean().item())

    if epoch%20 == 0:
        Gz_dist = G(z_fix.to(device)).detach().squeeze().cpu().numpy()
        Dx_ = D(grid_real.to(device)).detach().cpu().numpy()
        Gz_ = G(grid_z.to(device)).detach().cpu().numpy()

        ax = plt.subplot(221)
        plt.hist(z_fix.squeeze().numpy(), alpha=0.5, label='distrib(z)', \
        bins=20, density=True)
        plt.hist(Gz_dist, alpha=0.5, label='distrib(G(z))', \
        bins=20, density=True)
        plt.hist(real_fix.squeeze().numpy(), alpha=0.5, label='real',\
        bins=20, density=True)
        plt.plot(grid_real.numpy(), Dx_, 'b', label='D output')
        plt.ylim([0, 2])
        plt.legend()
        plt.title('epoch {}'.format(epoch))

        plt.subplot(223)
        plt.plot(grid_z.numpy(), Gz_, label='G(z)')
        plt.xlabel('z')
        plt.ylabel('G(z)')

        plt.subplot(122)
        plt.plot(D_reals, label='D(real)')
        plt.plot(D_fakes_1, label='D(G(z1))')
        plt.plot(D_fakes_2, label='D(G(z2))')
        plt.legend()
        
        plt.pause(0.2)
        plt.savefig(os.path.join(save_dir, 'e{}.jpg'.format(epoch)))
        plt.clf()

plt.close()