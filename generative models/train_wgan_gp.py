from __future__ import print_function
import argparse
import os
import pdb
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils import data
from torch import autograd

from tensorboardX import SummaryWriter
from tqdm import tqdm
from wgan_gp import Generator, Discriminator
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

dataset = dset.MNIST(root=opts.dataroot, download=True,
                    transform=transforms.Compose([
                        transforms.Resize(opts.imageSize),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                    ]))

assert dataset

# if you want to train on sub dataset
if opts.samples == -1:
    train_idx = range(len(dataset))
else:
    train_idx = range(0, len(dataset), len(dataset)//opts.samples)
    if len(train_idx) > opts.samples:
        train_idx = train_idx[:-1]

sampler = data.SubsetRandomSampler(train_idx)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts.batchSize, sampler=sampler,
                                         shuffle=False, num_workers=int(opts.workers))

device = torch.device("cuda" if opts.cuda else "cpu")

nz = opts.nz
ngf = opts.ngf
ndf = opts.ndf

G = Generator().to(device)
G.apply(weights_init)
if opts.G != '':
    G.load_state_dict(clean_state_dict(torch.load(opts.G)))

D = Discriminator().to(device)
D.apply(weights_init)
if opts.D != '':
    D.load_state_dict(clean_state_dict(torch.load(opts.D)))

if torch.cuda.device_count() > 1 and opts.cuda:
    print("Let's use {} GPUs".format(torch.cuda.device_count()))
    G = nn.DataParallel(G)
    D = nn.DataParallel(D)

# fixed noise
fixed_z = torch.randn(opts.batchSize, nz, 1, 1, device=device)

# setup optimizer
optimizerD = optim.Adam(D.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))

model_name = 'wgan_gp_{}samples_{}seed_{}ginner'.format(len(train_idx), opts.manualSeed, opts.ginner)
model_path = os.path.join('checkpoints', model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)
writer = SummaryWriter(os.path.join('runs', model_name))
n_iter = 0


def calc_gradient_penalty(netD, real_data, fake_data):
    BATCH_SIZE = real_data.shape[0]
    alpha = torch.rand(BATCH_SIZE, 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=None, only_inputs=True)[0]
    gradient_penalty = ((gradients.view(BATCH_SIZE, -1).norm(dim=1) - 1) ** 2).mean()
    return gradient_penalty

weight_gp = 10

for epoch in range(opts.epochs):
    print('[Epoch {}/{}]'.format(epoch+1, opts.epochs))
    for data in tqdm(dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        D.zero_grad()
        # train with real
        image = data[0].to(device)
        batch_size = image.size(0)
        output = D(image)
        D_x = output.mean()

        # train with fake
        z = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = G(z)
        output = D(fake.detach())
        D_G_z1 = output.mean()
        D_wgan = -D_x + D_G_z1
        
        gradient_penalty = calc_gradient_penalty(D, image, fake)
        D_wgan_gp = D_wgan + weight_gp * gradient_penalty
        D_wgan_gp.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        G.zero_grad()
        z = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = G(z)
        output = D(fake)
        D_G_z2 = output.mean()
        (-D_G_z2).backward()
        optimizerG.step()

        writer.add_scalar('D_wgan', D_wgan, n_iter)
        writer.add_scalar('D_wgan_gp', D_wgan_gp, n_iter)
        writer.add_scalar('D(x)', D_x, n_iter)
        writer.add_scalar('D(G(z1))', D_G_z1, n_iter)
        writer.add_scalar('D(G(z2))', D_G_z2, n_iter)

        if n_iter % 100 == 0:
            fake = G(fixed_z)
            real = vutils.make_grid(image, nrow=8, normalize=True, scale_each=True)
            writer.add_image('real', real, n_iter)
            fake = vutils.make_grid(fake.detach(), nrow=8, normalize=True, scale_each=True)
            writer.add_image('fake', fake, n_iter)

        n_iter += 1

    # do checkpointing
    torch.save(G.state_dict(), '%s/G_epoch_%d.pth' % (model_path, epoch))
    torch.save(D.state_dict(), '%s/D_epoch_%d.pth' % (model_path, epoch))