import os
from torch import load
from torchvision.utils import save_image
from models import Generator


def generate_image(fixed_noise, epoch):
    gen = Generator()
    gen.load_state_dict(load('./generator.pt'))

    images = gen(fixed_noise)

    if not os.path.exists('./generated_images'):
        os.mkdir('generated_images')

    image_save_path = "./generated_images/dcgan_epoch_{}/".format(epoch)
    for i, image in enumerate(images):
        if not os.path.exists(image_save_path):
            os.mkdir(image_save_path)
        save_image(image.data, image_save_path +
                   str(i) + ".png", normalize=True)
