import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--numImages', type=int, default=2000, help='number of images to use in training')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--numEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--continueTrain', type=bool, default=False, help='continue to train from saved dict')
parser.add_argument('--imageSize', type=int, default=128, help='side length of image')
parser.add_argument('--imageDims', type=tuple, default=(128, 128), help='dimensions of image')

parser.add_argument('--noiseSize', type=int, default=100, help='length of noise vector into generator')
parser.add_argument('--numChannels', type=int, default=3, help='number of input channels to discriminator (typically RGB)')
parser.add_argument('--featureMultiplier', type=int, default=64, help='size of feature maps in gen and disc')

parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta 1 value for optim')
parser.add_argument('--beta2', type=float, default=0.999, help='beta 2 value for optim')

parser.add_argument('--outputNum', type=int, default=10, help='images to output')

opts = parser.parse_args()
