from __future__ import division

import glob
import os
import re
import matplotlib.pyplot as pp
import numpy as np
import scipy.optimize
import scipy.misc

import skimage.data
from skimage import color, filter, io

from skimage import exposure, transform
#from skimage.filter import threshold_otsu, threshold_adaptive

from skimage import segmentation
from skimage.morphology import skeletonize


def preprocess(image, height=50, block_size=30):
    """Turn to greyscale, scale to a height, and then threshold to binary
    
    """
    
    image = color.rgb2grey(image)
    size_factor = float(height) / image.shape[0]
    new_size = [int(e*size_factor) for e in image.shape]
    image = transform.resize(image, new_size)
    image = filter.threshold_adaptive(image, block_size=30)
    
    return image
    
def separate_chars(image, n_chars=6):
    """Split the image vertically
    
    
    """
    
    
    image = np.asarray(image, dtype=np.int)
    width = image.shape[1]
    ideal_region_width = width / n_chars

    # floating point approximations to indices on the horizontal
    # axis of the image
    positions = np.asarray(np.linspace(0, width-1, num=n_chars+1), dtype=int)
    intensity = np.sum(image, axis=0)
    
    print image
    
    alpha = 0.5
    
    def objective(positions):
        # how much to the widths deviate from the ideal width
        # of the bins?
        positions = np.asarray(positions.clip(0, width-1), dtype=np.int)
        
        obj = alpha * np.sum(np.square((np.diff(positions) \
                              - ideal_region_width)))
                              
        # how many pixles are active along the verticle lines that
        # characterize the separatrixes between the characters?
        obj -= np.sum(intensity[positions])
        #print intensity[positions]
        return obj

    print positions, np.sum(intensity[positions])
    xf = scipy.optimize.fmin(objective, positions)
    xf = np.asarray(xf, dtype=np.int)

    chars = []
    for i in range(len(xf)-1):
        chars.append(image[:, xf[i]:xf[i+1]])

    return chars


def main():
    input_fn = glob.glob('data/*.png')
    #input_fn = ['./data/004018.png', './data/754298.png', './data/662458.png']
    fig, ax = pp.subplots(nrows=len(input_fn), ncols=6, figsize=(8, 5))
    pp.gray()

    for i, fn in enumerate(input_fn):
        final = preprocess(io.imread(fn))
        chars = separate_chars(final)
        for j, char in enumerate(chars):
            img = transform.resize(np.asarray(char, dtype=bool),
                output_shape=(25, 15))
            img = filter.threshold_adaptive(img, block_size=50)

            ax[i, j].imshow(img, interpolation='none')
            ax[i, j].axis('off')

    pp.suptitle('Binarized Camera Images')
    pp.savefig('binarized_camera.png')

if __name__ == '__main__':
    main()