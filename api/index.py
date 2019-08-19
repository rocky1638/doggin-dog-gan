from flask import Flask, request, send_file
import os
import json
from torch import load, randn
from torchvision import transforms
from torchvision.utils import save_image
from scripts.models import Generator


def send_image(img):
    save_image(img.data, "temp_dog.png", normalize=True)
    return send_file("temp_dog.png")


app = Flask(__name__)

generator = Generator()
generator.load_state_dict(load('./models/generator.pt', map_location="cpu"))


@app.route('/', methods=["GET"])
def index():
    return json.dumps({'success': True})


@app.route('/generate', methods=["GET"])
def generate():
    seed = randn(1, 100, 1, 1)
    image_tensor = generator(seed)[0]
    return send_image(image_tensor)


@app.route('/gannify', methods=["POST"])
def gannify():
    pass


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, host="0.0.0.0")
