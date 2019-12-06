import cv2
import sys
import glob
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv
from skimage.filters import threshold_otsu
# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median
from skimage.feature import canny

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 
    

def show_3d_image(img, title):
    fig = plt.figure()
    fig.set_size_inches((12,8))
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(0, img.shape[0], 1)
    Y = np.arange(0, img.shape[1], 1)
    X, Y = np.meshgrid(X, Y)
    Z = img[X,Y]

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 8)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title(title)
    plt.show()
    
def show_3d_image_filtering_in_freq(img, f):
    img_in_freq = fftpack.fft2(img)
    filter_in_freq = fftpack.fft2(f, img.shape)
    filtered_img_in_freq = np.multiply(img_in_freq, filter_in_freq)
    
    img_in_freq = fftpack.fftshift(np.log(np.abs(img_in_freq)+1))
    filtered_img_in_freq = fftpack.fftshift(np.log(np.abs(filtered_img_in_freq)+1))
    
    show_3d_image(img_in_freq, 'Original Image')
    show_3d_image(filtered_img_in_freq, 'Filtered Image')


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')
def ConvertToBinary(mPic,Threshold):
    for i in range(np.shape(mPic)[0]):
        for j in range (np.shape(mPic)[1]):
            if mPic[i][j] >= Threshold:
                mPic[i][j]=0
            else:
                mPic[i][j]=1
    return mPic

def ANDING_SE(Fore,Back):
    for i in range(np.shape(Fore)[0]):
        for j in range (np.shape(Fore)[1]):
            if Back[i][j]==1:
                if Fore[i][j] != Back[i][j]:
                    return 0
    return 1
def ORI_SE(Fore,Back):
    for i in range(np.shape(Fore)[0]):
        for j in range (np.shape(Fore)[1]):
            if Back[i][j]==1:
                if Fore[i][j] == Back[i][j]:
                    return 1
    return 0
    
def mErosion(mPic,SE):
    WindowWidth=np.shape(SE)[0]
    WindowHeight=np.shape(SE)[1]
    mOut=np.zeros((np.shape(mPic)[0],np.shape(mPic)[1]))
    EdgeX=(int)(WindowWidth/2)
    EdgeY=(int)(WindowHeight/2)
    for i in range(np.shape(mPic)[0]-EdgeX):
        for j in range (np.shape(mPic)[1]-EdgeY):
            foreground =np.zeros((WindowWidth,WindowHeight))
            for fx in range(WindowWidth):
                for fy in range (WindowHeight):
                    foreground[fx][fy]=mPic[i+fx-EdgeX][j+fy-EdgeY]
            mOut[i][j]=ANDING_SE(foreground,SE)
    return mOut
    
def mDilation(mPic,SE):
    WindowWidth=np.shape(SE)[0]
    WindowHeight=np.shape(SE)[1]
    mOut=np.zeros((np.shape(mPic)[0],np.shape(mPic)[1]))
    EdgeX=(int)(WindowWidth/2)
    EdgeY=(int)(WindowHeight/2)
    for i in range(np.shape(mPic)[0]-EdgeX):
        for j in range (np.shape(mPic)[1]-EdgeY):
            foreground =np.zeros((WindowWidth,WindowHeight))
            for fx in range(WindowWidth):
                for fy in range (WindowHeight):
                    foreground[fx][fy]=mPic[i+fx-EdgeX][j+fy-EdgeY]
            mOut[i][j]=ORI_SE(foreground,SE)
    return mOut

def PrintBinary(PIC):
    io.imshow(PIC,cmap="binary") # 0~255 np.zeros((2, 1))
    io.show()
def my_erosion(img, mask):
    shape = img.shape
    new_img = np.copy(img)
    out = int(np.floor(len(mask) / 2))
    for i in range(out, shape[0] - out):
        for j in range(out, shape[1] - out):
            portion = img[i - out:i + out + 1, j - out:j + out + 1]
            mat = np.multiply(mask, portion)
            new_img[i, j] = np.min(mat)
    return new_img


def my_dilation(img, mask):
    shape = img.shape
    new_img = np.copy(img)
    out = int(np.floor(len(mask) / 2))
    for i in range(out, shape[0] - out):
        for j in range(out, shape[1] - out):
            portion = img[i - out:i + out + 1, j - out:j + out + 1]
            mat = np.multiply(mask, portion)
            new_img[i, j] = np.max(mat)
    return new_img


def Opening(mPic, SE):
    return my_dilation(my_erosion(mPic, SE), SE)


def Closing(mPic, SE):
    return my_erosion(my_dilation(mPic, SE), SE)