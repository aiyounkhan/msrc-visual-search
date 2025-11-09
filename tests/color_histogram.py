import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def grayHistogram():
    img_path ='/Users/akanady/Desktop/CVPR_Assignment_1/tests/Screenshot 2025-11-09 at 02.09.04.png'
    img = cv.imread(img_path, cv.IMREAD_REDUCED_GRAYSCALE_2)

    plt.figure()
    plt.imshow(img, cmap='gray')
    hist = cv.calcHist([img], [0], None, [256], [0, 256])

    plt.figure()
    plt.plot(hist)
    plt.xlabel('bins')
    plt.ylabel('# of pixels')

    plt.show()

def colorHistogram():
    img_path ='/Users/akanady/Desktop/CVPR_Assignment_1/tests/Screenshot 2025-11-09 at 02.09.04.png'
    img = cv.imread(img_path)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(imgRGB)

    colors = ['b', 'g', 'r']

    plt.figure()
    for i in range(len(colors)):
        hist = cv.calcHist([imgRGB], [i], None, [256], [0, 256])
        plt.plot(hist, colors[i])

    plt.xlabel('pixel intensity')
    plt.ylabel('# of pixels')
  
    plt.show()

def histogramRegion():
    img_path ='/Users/akanady/Desktop/CVPR_Assignment_1/tests/Screenshot 2025-11-09 at 02.09.04.png'
    img = cv.imread(img_path)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgRGB = imgRGB[100:200, 200:300, :]

    plt.figure()
    plt.imshow(imgRGB)

    colors = ['b', 'g', 'r']

    plt.figure()
    for i in range(len(colors)):
        hist = cv.calcHist([imgRGB], [i], None, [256], [0, 256])
        plt.plot(hist, colors[i])

    plt.xlabel('pixel intensity')
    plt.ylabel('# of pixels')
  
    plt.show()


if __name__ == '__main__':
    # grayHistogram()
    # colorHistogram()
    histogramRegion()

