import os
import sys
from PIL import Image
import shutil
from imageio import imread
import numpy as np
import torch
import torchvision.transforms as transforms
from test_config import config
from config import network_config
from werkzeug.utils import secure_filename
from flask import Flask, render_template, flash, request, redirect, url_for

app = Flask(__name__)
app.debug = True
network = None
model_path = 'saved_model/299.pth.tar'
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_model(args):
    network, _ = network_config(args,'test', None, True, model_path, False)
    network.eval()

def retrieve_captions():
    return ['temp/image.jpg' for i in range(21)]

def retrieve_images(caption):
    return ['temp/image.jpg' for i in range(21)]

def get_query(request):
    try:
        text = request.form['textquery']
    except:
        text = None
    
    try:
        image = request.files['imagequery']
        image.save('static/temp/image.png')
    except:
        image = None

    if text is None:
        return (image, 'image')
    else:
        return (text, 'text')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict_from_location():
    query, query_type = get_query(request)
    if query_type == 'text':
        retrieved_images = retrieve_images(query)
        return render_template('image_results.html', data=retrieved_images)
    else:
        retrieved_captions = retrieve_captions()
        return render_template('text_results.html', data=retrieved_captions)

if __name__ == '__main__':
    print('Parsing arguments...')
    args = config()
    print('Loading model weights...')
    load_model(args)
    app.run()