import os
from torch import load, from_numpy, FloatTensor
from torchvision.utils import save_image
import numpy as np
from models import Generator

gen = Generator()
gen.load_state_dict(load('./generator.pt'))

seed = from_numpy(np.random.normal(0,1, size=(1, 100))).type(FloatTensor)
images = gen(seed)

if not os.path.exists('./generated_images'):
	os.mkdir('generated_images')

save_image(images.data, "./generated_images/test.png", normalize=True)