'''Create image sprite'''

import os
import numpy as np
import glob
from PIL import Image
from keras.preprocessing import image


#file_dir = "/home/bisrat/flower_photos/" # path to row data set
#files = os.listdir(file_dir)
#filelist = glob.glob(file_dir + '*.jpg')

def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    #data = data.transpose(0, 2, 3, 1)
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data

def create_sprite(file_dir):
    filelist = glob.glob(file_dir + '*.jpg')
    w, h = (320, 320)

    img_array = []

    for fpath in filelist:
        img = image.load_img(fpath, target_size=(w, h))
        x = image.img_to_array(img)
        img_array.append(x)

    img_sprite = images_to_sprite(np.asarray(img_array))
    sprite_path = os.path.join(file_dir, 'sprite.png')
    img = Image.fromarray(img_sprite, 'RGB')
    img = img.resize((8192, 8192), Image.ANTIALIAS)
    img.save(sprite_path)


