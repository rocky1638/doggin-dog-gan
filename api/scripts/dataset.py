import glob
from torch.utils import data
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
from options import opts

DATASET_MEAN = (0.5, 0.5, 0.5)
DATASET_STD = (0.5, 0.5, 0.5)


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, image_folder_path, evalu):
        'Initialization'
        self.evalu = evalu
        image_paths = glob.glob('{}/*'.format(image_folder_path))
        self.datalist = [
            image_path for image_path in image_paths][:opts.numImages]
        self.std = 0

    def __len__(self):
        'Denotes the total number of samples'
        return opts.numImages
        # return len(self.datalist)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data
        real_image = Image.open(self.datalist[index]).convert('RGB')

        # crop here
        side_length = min(real_image.size)
        image_transformation = transforms.Compose([
            transforms.RandomCrop(size=side_length),
            transforms.Resize(size=opts.imageDims,
                              interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ])

        real_image = image_transformation(real_image)

        # adding noise to ensure that discriminator does not overfit immediately
        noise = torch.from_numpy(np.random.normal(
            0, self.std, size=opts.imageDims)).type(torch.FloatTensor)
        real_image += noise
        return real_image


def GenerateIterator(image_path, evalu=False, shuffle=True):
    params = {
        'batch_size': opts.batchSize,
        'shuffle': shuffle,
        'num_workers': 8,
        'pin_memory': False,
        'drop_last': False,
    }

    return data.DataLoader(Dataset(image_path, evalu=evalu), **params)
