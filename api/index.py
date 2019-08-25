import os
import io
import base64
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
    myImage = open('temp_dog.png', 'rb')
    myBase64File = base64.b64encode(myImage.read()).decode('ascii')
    return myBase64File


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
    return json.dumps({'image': send_image(image_tensor)})


@app.route('/gannify', methods=["POST"])
def gannify():
    data = request.data
    dataDict = json.loads(data)
    base64dog = dataDict['dog']
    rawdog = base64.b64decode(base64dog)
    dog = Image.open(io.BytesIO(rawdog))

    cropped_dog = randomCrop(dog)
    encoded_dog = encoder(cropped_dog)
    gannified_dog = generator(encoded_dog)[0]

    return json.dumps({'image': send_image(gannified_dog)})


@app.route('/images', methods=["GET"])
def images():
    index = random.randint(1, 10)
    filename = str(index) + ".jpg"
    myImage = open('./online_dogs/' + filename, 'rb')
    myBase64File = base64.b64encode(myImage.read()).decode('ascii')
    return json.dumps({'image': myBase64File})


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, host="0.0.0.0")
