import os
import json
import random
from flask import Flask, request, send_file
from PIL import Image
import numpy as np
from torch import load, randn
from torchvision import transforms
from torchvision.utils import save_image
from scripts.models import Generator, Imageencoder


def send_image(img):
    save_image(img.data, "temp_dog.png", normalize=True)
    return send_file("temp_dog.png")


def randomCrop(img):
    width, height = img.size
    img = np.asarray(img)
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = Image.fromarray(img[y:y+height, x:x+width])
    img = img.resize((128, 128), resample=Image.NEAREST)
    img_tensor = transforms.ToTensor()(img)
    return transforms.Normalize(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    )(img_tensor).unsqueeze(0)


app = Flask(__name__)

generator = Generator()
generator.load_state_dict(load('./models/generator.pt', map_location="cpu"))
encoder = Imageencoder()
encoder.load_state_dict(load('./models/encoder.pt', map_location="cpu"))


@app.route('/', methods=["GET"])
def index():
    return json.dumps({'success': True})


@app.route('/generate', methods=["GET"])
def generate():
    seed = randn(1, 100, 1, 1)
    image_tensor = generator(seed)[0]
    return send_image(image_tensor)


@app.route('/gannify', methods=["GET"])
def gannify():
    num = random.randint(1, 3)
    dog = Image.open(f"./online_dogs/{num}.jpg")
    cropped_dog = randomCrop(dog)
    encoded_dog = encoder(cropped_dog)
    gannified_dog = generator(encoded_dog)[0]
    return send_image(gannified_dog)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, host="0.0.0.0")
