# original paper: https://arxiv.org/abs/1706.08500
# orignal code [tf]: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
# mostly copy from [pytorch]: https://github.com/mseitzer/pytorch-fid

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
import numpy as np
from scipy import linalg
from tqdm import tqdm

class FIDModel(nn.Module):
    """The FID is calculated by assuming that X_1 and X_2 are 
    the activations of the pool_3 layer of the inception net 
    for generated samples and real world samples respectively.
    
    Note: We have to re-extract the blocks because the original PyTorch
    InceptionV3 model using nn.functional APIs in the forward 
    method, these pooling APIs from nn.functional has to be added
    to the blocks manually.
    """
    def __init__(self):
        super(FIDModel, self).__init__()
        inception = inception_v3(pretrained=True, aux_logits=True)
        blocks = []
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        blocks.extend(block0)

        block1 = [
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        blocks.extend(block1)

        block2 = [
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        ]
        blocks.extend(block2)

        block3 = [
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        ]
        blocks.extend(block3)
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        # [-1.0, 1.0] --> [0, 1.0]
        x = x * 0.5 + 0.5
        # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        # --> mean = 0, std = 1
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        #
        # --> fixed-size input: batch x 3 x 299 x 299
        # x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        return self.blocks(x).view(x.shape[0], -1)


def compute_fid(real_preds, fake_preds):
    """
    Computes Frechet inception distance (FID) between real and fake predictions
    real -- Tensor of [Nr, 2048]
    fake -- Tensor of [Nf, 2048]
    """
    real_np = real_preds.detach().cpu().numpy()
    fake_np = fake_preds.detach().cpu().numpy()
    # import pdb; pdb.set_trace()
    mu1, mu2 = [np.mean(preds, axis=0) for preds in [real_np, fake_np]]
    sigma1, sigma2 = [np.cov(preds, rowvar=False) for preds in [real_np, fake_np]]
    print('[FID] => Compute FID (about 15 seconds on my MBP 2012, Core i5 2.5GHz) ...')
    # The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    # and X_2 ~ N(mu_2, C_2) is
    # d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    diff = mu1 - mu2
    # Product might be almost singular, we have to care numerical issue
    eps=1e-6
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('[FID] => fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

if __name__ == '__main__':
    # real = torch.randn(20, 3, 32, 32)
    # real_dataset = torch.utils.data.TensorDataset(real)

    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='./cifar10', download=True,
                             transform=transforms.Compose([
                                 transforms.Resize([32,32]),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
    )
    # only use the first 20 images for demonstration
    cifar = torch.utils.data.Subset(cifar, range(20))
    IgnoreLabelDataset(cifar)
    real = IgnoreLabelDataset(cifar)

    fake = torch.tanh(torch.randn(15, 3, 32, 32))
    fake_dataset = torch.utils.data.TensorDataset(fake)
    print(compute_fid(real, fake))