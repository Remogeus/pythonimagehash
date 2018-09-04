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

import sys, getopt
import numpy as np

def imgPrepare(filename):
    """ imgPrepare(filename)
        
        Returns a numpy array created from grayscale 32x32 version of the image
        Parameters: @filename - name of the image that should be returned

        Returns: numpy array
    """
    image = Image.open(filename).convert('LA').resize((32,32))
    return np.asarray(image)

def dct2d(arg):
    """ dct2d(arg)

        Returns a result of 2D DCT (discrete cosine transform)
        Parameters: @arg - numpy array used for calculation

        Returns: numpy array
    """
    return dct(dct(arg.T, norm='ortho').T, norm='ortho')

def imgHashCreate(filename):
    imgArr = imgPrepare(filename)
    imgArr = imgArr[:, :, :1].transpose(2,0,1)
    imgArr = np.squeeze(imgArr)
    imgDCT = dct2d(imgArr)
    imgDCT = imgDCT[:8,:8]
    imgDCTAvg = np.mean(imgDCT[1:,:])
    imgHash = [0]*64
    imgDCT = imgDCT.flatten()
    for x in range(0,64):
        if imgDCT[x] > imgDCTAvg:
            imgHash[x] = 1
        else:
            imgHash[x] = 0
    return ''.join(map(str,imgHash))
    
def hamming2(s1,s2):
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1,s2))

if __name__=='__main__':
    image1 = ''
    image2 = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hf:s:",["fstimage=","sndimage="])
    except getopt.GetoptError:
        print('usage: imagehash.py -f <firstImage> -s <secondImage>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
           print('usage: imagehash.py -f <firstImage> -s <secondImage>') 
           sys.exit()
        elif opt in ("-f","--fstimage"):
           fstImage = arg
        elif opt in ("-s", "--sndimage"):
           sndImage = arg

    imgHash1 = imgHashCreate(fstImage)
    imgHash2 = imgHashCreate(sndImage)
    print(fstImage, end = "\t", flush=True)
    print(hex(int(imgHash1,2)), end = "\t", flush=True)
    print(sndImage, end = "\t", flush=True)
    print(hex(int(imgHash2,2)), end = "\t", flush=True)
    print(hamming2(imgHash1, imgHash2))
    

