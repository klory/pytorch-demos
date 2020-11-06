import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='data', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--samples', type=int, default=-1)
    parser.add_argument('--nG', type=int, default=1)

    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--nc', type=int, default=1)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)

    parser.add_argument('--G', default='', help="path to G (to continue training)")
    parser.add_argument('--D', default='', help="path to D (to continue training)")

    return parser

