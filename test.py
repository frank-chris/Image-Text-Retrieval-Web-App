import os
import sys
from PIL import Image
from imageio import imread
import numpy as np
import torch
import torchvision.transforms as transforms
from test_config import config
from config import network_config

model_path = 'saved_model/299.pth.tar'
img_path = '12.jpg'

def main(args):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    network, _ = network_config(args,'test', None, True, model_path, False)
    network.eval()
    captions = torch.tensor([[348, 349,   8,  19,   6,  46,   5,  44,   4,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0], 
          [348, 349,   8,  19,   6,  46,   5,  44,   4,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0]])
    captions_length = torch.tensor([9, 9])
    img = imread(img_path)
    img = np.array(Image.fromarray(img).resize(size=(224,224)))
    images = test_transform(img)
    images = torch.reshape(images, (1, images.shape[0], images.shape[1], images.shape[2]))
    with torch.no_grad():
        image_embeddings, text_embeddings = network(images, captions, captions_length)

    print(image_embeddings.shape)
    print(text_embeddings.shape)

if __name__ == '__main__':
    args = config()
    main(args)


