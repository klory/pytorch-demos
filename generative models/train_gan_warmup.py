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

from torch.utils.tensorboard import SummaryWriter
from gan import Generator, Discriminator
from utils import clean_state_dict, weights_init
from tqdm import tqdm

from args import get_parser
# =============================================================================
parser = get_parser()
args = parser.parse_args()
# =============================================================================

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset = dset.MNIST(root=args.dataroot, download=True,
                    transform=transforms.Compose([
                        transforms.Resize(args.imageSize),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                    ]))

assert dataset

# if you want to train on sub dataset
if args.samples == -1:
    train_idx = range(len(dataset))
else:
    train_idx = range(0, len(dataset), len(dataset)//args.samples)
    if len(train_idx) > args.samples:
        train_idx = train_idx[:-1]

sampler = data.SubsetRandomSampler(train_idx)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, sampler=sampler,
                                         shuffle=True, num_workers=int(args.workers))

device = torch.device("cuda" if args.cuda else "cpu")

nz = args.nz
ngf = args.ngf
ndf = args.ndf

G = Generator().to(device)
G.apply(weights_init)
if args.G != '':
    G.load_state_dict(clean_state_dict(torch.load(args.G)))

D = Discriminator().to(device)
D.apply(weights_init)
if args.D != '':
    D.load_state_dict(clean_state_dict(torch.load(args.D)))

if torch.cuda.device_count() > 1 and args.cuda:
    print("Let's use {} GPUs".format(torch.cuda.device_count()))
    G = nn.DataParallel(G)
    D = nn.DataParallel(D)

criterion = nn.BCELoss()

# fixed noise & label
fixed_noise = torch.randn(args.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

model_name = f'gan_warmup_samples{len(train_idx)}_seed{args.manualSeed}_nG{args.nG}'
save_dir = os.path.join('runs', model_name)
writer = SummaryWriter(save_dir)
n_iter = 0

for epoch in range(args.epochs):
    pbar = tqdm(dataloader)
    for data in pbar:
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        D.zero_grad()
        image = data[0].to(device)
        batch_size = image.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = D(image)
        errD_real = criterion(output, label)
        D_x = output.mean().item()

        # train with fake
        z = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = G(z)
        # label.fill_(fake_label)
        label = torch.full((batch_size,), fake_label, device=device)
        output = D(fake.detach())
        errD_fake = criterion(output, label)
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        for g_iter in range(args.nG):
            G.zero_grad()
            z = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = G(z)
            label.fill_(real_label)  # fake labels are real for generator cost
            output = D(fake)
            errG = criterion(output, label)
            D_G_z2 = output.mean().item()
            errG.backward()
            optimizerG.step()

        # state_msg = f'{epoch+1:d}'
        state_msg = 'Epoch: %d Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' \
              % (epoch, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)
        
        pbar.set_description(state_msg)

        writer.add_scalar('loss_D', errD, n_iter)
        writer.add_scalar('loss_G', errG, n_iter)
        writer.add_scalar('D(x)', D_x, n_iter)
        writer.add_scalar('D(G(z1))', D_G_z1, n_iter)
        writer.add_scalar('D(G(z2))', D_G_z2, n_iter)
        if n_iter % 100 == 0:
            fake = G(fixed_noise)
            real = vutils.make_grid(image, nrow=8, normalize=True, scale_each=True)
            writer.add_image('real', real, n_iter)
            fake = vutils.make_grid(fake.detach(), nrow=8, normalize=True, scale_each=True)
            writer.add_image('fake', fake, n_iter)

        n_iter += 1

    # do checkpointing
    torch.save(G.state_dict(), f'{save_dir}/G_epoch_{epoch}.pth')
    torch.save(D.state_dict(), f'{save_dir}/D_epoch_{epoch}.pth')