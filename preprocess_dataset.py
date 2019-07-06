# coding=utf8

"""
"""

import os
import numpy as np
import imageio
from skimage import io, exposure
import argparse
import sys
from tqdm import tqdm


ROOT = '/home/sedigheh/lung_segmentation/dataset/JSRT/JSRT'
# ROOT = '/home/sedigheh/lung_segmentation/dataset/CT-kaggle-lung/2d_images'
# ROOT = '/home/sedigheh/lung_segmentation/dataset/MontgomerySet/CXR_png'


def make_lungs():
    for i, filename in tqdm(enumerate(os.listdir(ROOT))):
        if(filename != "preprocessed"):
            input_path = os.path.join(ROOT, filename)
            img = imageio.imread(input_path)
            img = 1 - imageio.imread(input_path) * 1/255
            # img = imageio.imread(input_path).reshape(2048, 2048)
            # img = imageio.imread(input_path).reshape(512, 512, 3)
            img = imageio.imread(input_path).reshape(512, 512)
            img = exposure.equalize_adapthist(img)
            output_path = os.path.join(ROOT, 'preprocessed', filename)
            io.imsave(output_path, img)
            print('Lung', i+1, filename)


def make_masks():
    path = '/path/to/JSRT/All247images/'
    for i, filename in enumerate(os.listdir(path)):
        left = io.imread('/path/to/JSRT/Masks/left lung/' +
                         filename[:-4] + '.gif')
        right = io.imread('/path/to/JSRT/Masks/right lung/' +
                          filename[:-4] + '.gif')
        io.imsave('/path/to/JSRT/new/' + filename[:-4] +
                  'msk.png', np.clip(left + right, 0, 255))
        print('Mask', i, filename)


def main(mode):
    if (mode == "masks") or (mode == "mask"):
        make_masks()
    elif (mode == "lungs") or (mode == "lung"):
        make_lungs()
    else:
        print("Mode not supported!")
        sys.exit()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess images")
    parser.add_argument("--mode",
                        help="Supported modes are lung(s) and mask(s). " +
                        "Defaults to lungs.",
                        default="lungs")
    args = parser.parse_args()
    main(args.mode)
