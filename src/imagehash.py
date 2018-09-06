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

def imgPrepare(filename):
    """ imgPrepare(filename)
        
        Returns a numpy array created from grayscale 32x32 version of the
        image. Grayscaling is achieved by using the LA (grayscale with
        alpha) mode from PIL and is done for simplifying the subsequent 
        calculations (same with downscaling the images to 32x32 pixels)

        Parameters: @filename - string, name of the image that should be returned
        Returns:    numpy array
    """
    image = Image.open(filename).convert('LA').resize((32,32))
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

def imgHashCreate(filename):
    """imgHashCreate(filename)

        Returns a hash created from an image using the pHash algorithm
        (perceptual hashing) using DCT.

        Parameters: @filename - string, name of the image from which 
                    the hash is calculated
        Returns:    string - 64 bit number
    """
    imgArr = imgPrepare(filename)
    imgArr = np.squeeze(imgArr[:, :, :1].transpose(2,0,1))
    imgDCT = dct2d(imgArr)
    imgDCT = imgDCT[:8,:8]
    imgDCTAvg = np.mean(imgDCT[1:,:])
    imgHash = [0]*64
    imgDCT = imgDCT.flatten()
    for x in range(0,64):
        if imgDCT[x] > imgDCTAvg:
            imgHash[x] = 1
    return ''.join(map(str,imgHash))
    
def hamming2(s1,s2):
    """ hamming2(s1, s2)
        Returns Hamming distance calculated from 2 binary numbers. Hamming
        distance measures the minimum amount of substitutions required to
        change one string to another string of equal lenght.

        Parameters: @s1, @s2 - string, binary number representing (in this
                    case) 2 hashes of 2 images
        Returns:    integer - Hamming distance
    """
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1,s2))

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog='imagehash',
        description='The script creates hashes from two images and then '
            'calculates their Hamming distance to measure diff'
            'erence between them'
            )
    parser.add_argument(
        'firstImage',
        metavar = 'firstImage',
        nargs = 1,
        help = 'first image to be processed'
            )
    parser.add_argument(
        'secndImage',
        metavar = 'secndImage',
        nargs = 1,
        help = 'second image to be processed'
            )
    
    args = parser.parse_args()
    imgHash1 = imgHashCreate(os.path.join(local_folder, ''.join(args.firstImage)))
    imgHash2 = imgHashCreate(os.path.join(local_folder, ''.join(args.secndImage)))
    print(''.join(args.firstImage), end = "\t", flush=True)
    print(hex(int(imgHash1,2)), end = "\t", flush=True)
    print(''.join(args.secndImage), end = "\t", flush=True)
    print(hex(int(imgHash2,2)), end = "\t", flush=True)
    print(hamming2(imgHash1, imgHash2))
    

