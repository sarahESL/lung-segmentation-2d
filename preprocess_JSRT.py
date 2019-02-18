# coding=utf8

"""
"""

import os
import numpy as np
import imageio
from skimage import io, exposure


ROOT = '/home/sedigheh/lung_playground/dataset/JSRT/augmented/AugmentedImages/512_Originals_Augmented'


def make_lungs():
    for i, filename in enumerate(os.listdir(ROOT)):
        if(filename != "preprocessed"):
            print(filename)
            input_path = os.path.join(ROOT, filename)
            img = 1 - imageio.imread(input_path) * 1/255
            # img = imageio.imread(input_path).reshape(2048, 2048)
            img = imageio.imread(input_path).reshape(512, 512, 3)
            img = exposure.equalize_adapthist(img)
            output_path = os.path.join(ROOT, 'preprocessed', filename)
            io.imsave(output_path, img)
            print('Lung', i+1, filename)

def make_masks():
    path = '/path/to/JSRT/All247images/'
    for i, filename in enumerate(os.listdir(path)):
        left = io.imread('/path/to/JSRT/Masks/left lung/' + filename[:-4] + '.gif')
        right = io.imread('/path/to/JSRT/Masks/right lung/' + filename[:-4] + '.gif')
        io.imsave('/path/to/JSRT/new/' + filename[:-4] + 'msk.png', np.clip(left + right, 0, 255))
        print('Mask', i, filename)

make_lungs()
#make_masks()
