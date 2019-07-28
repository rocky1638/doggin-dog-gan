import os
from torch import load, from_numpy, FloatTensor
from torchvision.utils import save_image
import numpy as np
from models import Generator

def generate_image(generator_path, epoch, num_images):	
    gen = Generator()
    gen.load_state_dict(load('./generator.pt'))

    seed = from_numpy(np.random.normal(0,1, size=(num_images, 100))).type(FloatTensor)
    images = gen(seed)

    if not os.path.exists('./generated_images'):
        os.mkdir('generated_images')

    for i in range(num_images):
        image_save_path = "./generated_images/epoch_" + str(epoch) + "_" + str(i+1) + ".png"
        save_image(images[i].data, image_save_path, normalize=True)