"""
MIT License

Copyright (c) 2018 Ondrej Povolny

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from scipy.fftpack import dct
from PIL import Image

import sys
import argparse
import numpy as np
import os


local_folder = os.path.dirname(os.path.abspath(__file__))


def img_prepare(filename):
    """ img_prepare(filename)

        Returns a numpy array created from grayscale 32x32 version of the
        image. Grayscaling is achieved by using the LA (grayscale with
        alpha) mode from PIL and is done for simplifying the subsequent
        calculations (same with downscaling the images to 32x32 pixels)

        Parameters: @filename - string, name of the image that should be
                    returned
        Returns:    numpy array
    """
    image = Image.open(filename).convert('LA').resize((32, 32))
    return np.asarray(image)


def dct2d(arg):
    """ dct2d(arg)

        Returns a result of 2D DCT (discrete cosine transform). DCT is simillar
        to Fourier transform, but unlike DFT, results of DCT are always real
        numbers and the transform only uses cosine function.

        Parameters: @arg - numpy array, argument used for calculation
        Returns:    numpy array
    """
    return dct(dct(arg.T, norm='ortho').T, norm='ortho')


def img_hash_create(filename):
    """img_hash_create(filename)

        Returns a hash created from an image using the pHash algorithm
        (perceptual hashing) using DCT.

        Parameters: @filename - string, name of the image from which
                    the hash is calculated
        Returns:    string - 64 bit number
    """
    img_arr = img_prepare(filename)
    img_arr = np.squeeze(img_arr[:, :, :1].transpose(2, 0, 1))
    img_dct = dct2d(img_arr)
    img_dct = img_dct[:8, :8]
    img_dct_avg = np.mean(img_dct[1:, :])
    img_hash = [0]*64
    img_dct = img_dct.flatten()
    for x in range(0, 64):
        if img_dct[x] > img_dct_avg:
            img_hash[x] = 1
    return ''.join(map(str, img_hash))


def hamming_bin(s1, s2):
    """ hamming_bin(s1, s2)
        Returns Hamming distance calculated from 2 binary numbers. Hamming
        distance measures the minimum amount of substitutions required to
        change one string to another string of equal lenght.

        Parameters: @s1, @s2 - string, binary number representing (in this
                    case) 2 hashes of 2 images
        Returns:    integer - Hamming distance
    """
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='imagehash',
        description='The script creates hashes from two images and then '
                    'calculates their Hamming distance to measure diff'
                    'erence between them'
            )
    parser.add_argument(
        'first_image',
        metavar='first_image',
        nargs=1,
        help='first image to be processed'
            )
    parser.add_argument(
        'second_image',
        metavar='second_image',
        nargs=1,
        help='second image to be processed'
            )

    args = parser.parse_args()
    img_hash_1 = img_hash_create(
            os.path.join(local_folder, ''.join(args.first_image)))
    img_hash_2 = img_hash_create(
            os.path.join(local_folder, ''.join(args.second_image)))
    print(''.join(args.first_image), end="\t", flush=True)
    print(hex(int(img_hash_1, 2)), end="\t", flush=True)
    print(''.join(args.second_image), end="\t", flush=True)
    print(hex(int(img_hash_2, 2)), end="\t", flush=True)
    print(hamming_bin(img_hash_1, img_hash_2))
