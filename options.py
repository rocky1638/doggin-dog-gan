import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--numImages', type=int, default=0, help='number of images to use in training')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--numEpochs', type=int, default=200, help='number of epochs to train for')

opts = parser.parse_args()