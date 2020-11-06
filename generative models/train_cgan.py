from __future__ import print_function
import argparse
import os
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

from tensorboardX import SummaryWriter
from cgan import Generator, Discriminator
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

if opts.samples == -1:
    train_idx = range(len(dataset))
else:
    train_idx = range(0, len(dataset), len(dataset)//opts.samples)

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

criterion = nn.BCELoss()

# fixed noise & label
temp_z_ = torch.randn(10, 100)
fixed_z_ = temp_z_
fixed_y_ = torch.zeros(10, 1)
for i in range(9):
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
    temp = torch.ones(10, 1) + i
    fixed_y_ = torch.cat([fixed_y_, temp], 0)

fixed_z_ = fixed_z_.view(-1, 100, 1, 1) # [100, 100, 1, 1]
fixed_y_label_ = torch.zeros(100, 10)
fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
fixed_y_label_ = fixed_y_label_.view(-1, 10, 1, 1)

fixed_z_ = fixed_z_.to(device) # [100, 100, 1, 1]
fixed_y_label_ = fixed_y_label_.to(device) # [100, 10, 1, 1]

# label preprocess
onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1,
    torch.LongTensor(range(10)).view(10, 1), 1).view(10, 10, 1, 1) 
# onehot:
# [[1,0,0,0,0,0,0,0,0],
#  [0,1,0,0,0,0,0,0,0],
#  [0,0,1,0,0,0,0,0,0],
# ...
#  [0,0,0,0,0,0,0,0,1]]

fill = torch.zeros([10, 10, opts.imageSize, opts.imageSize])
for i in range(10):
    fill[i, i, :, :] = 1
# fill: there are ten image mask, each mask is a 10-dim image, with only one dim has all ones.

# setup optimizer
optimizerD = optim.Adam(D.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))

model_name = '{}samples_{}seed_{}ginner'.format(len(train_idx), opts.manualSeed, opts.ginner)
model_path = os.path.join('checkpoints', model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)
writer = SummaryWriter(os.path.join('runs', model_name))
n_iter = 0

for epoch in range(opts.epochs):
    for i, data in enumerate(dataloader, 0):
        image, label = data
        batch_size = image.shape[0]
        y_real = torch.ones(batch_size).to(device)
        y_fake = torch.zeros(batch_size).to(device)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        D.zero_grad()
        # train with real
        image = image.to(device)
        y_fill = fill[label].to(device) # [B, 10, H, W]
        output = D(image, y_fill).squeeze() # [B]
        D_x = output.mean().item()
        errD_real = criterion(output, y_real)

        # train with fake
        z = torch.randn(batch_size, 100).view(-1, 100, 1, 1) # [B, 100, 1, 1]
        y = (torch.rand(batch_size, 1) * 10).long().squeeze() # [B], arbitrary labels
        y_label = onehot[y] # [B, 10, 1, 1]
        y_fill = fill[y]
        z, y_label, y_fill = z.to(device), y_label.to(device), y_fill.to(device)
        fake = G(z, y_label)
        output = D(fake.detach(), y_fill).squeeze()
        D_G_z1 = output.mean().item()
        errD_fake = criterion(output, y_fake)
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()
        
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        G.zero_grad()
        z = torch.randn(batch_size, 100).view(-1, 100, 1, 1) # [B, 100, 1, 1]
        y = (torch.rand(batch_size, 1) * 10).long().squeeze() # [B], arbitrary labels
        y_label = onehot[y] # [B, 10, 1, 1]
        y_fill = fill[y]
        z, y_label, y_fill = z.to(device), y_label.to(device), y_fill.to(device)
        fake = G(z, y_label)
        output = D(fake, y_fill).squeeze()
        D_G_z2 = output.mean().item()
        errG = criterion(output, y_real)
        errG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opts.epochs-1, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        writer.add_scalar('loss_D', errD, n_iter)
        writer.add_scalar('loss_G', errG, n_iter)
        writer.add_scalar('D(x)', D_x, n_iter)
        writer.add_scalar('D(G(z1))', D_G_z1, n_iter)
        writer.add_scalar('D(G(z2))', D_G_z2, n_iter)
        if n_iter % 100 == 0 or i == len(dataloader)-1:
            real_image = vutils.make_grid(image, nrow=8, normalize=True, scale_each=True)
            writer.add_image('real', real_image, n_iter)
            
            fake = G(fixed_z_, fixed_y_label_)
            fake_image = vutils.make_grid(fake.detach(), nrow=10, normalize=True, scale_each=True)
            writer.add_image('fake', fake_image, n_iter)

        n_iter += 1

    # do checkpointing
    torch.save(G.state_dict(), '%s/netG_epoch_%d.pth' % (model_path, epoch))
    torch.save(D.state_dict(), '%s/netD_epoch_%d.pth' % (model_path, epoch))