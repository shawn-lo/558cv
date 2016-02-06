import numpy as np
import scipy
from scipy import ndimage
from PIL import Image
#from pylab import *

class Filters():
    def gaussian_filtering(self, befIm, sigma):
        #winSize = sigma*4 + 1
        winSize = sigma*6 - 1
        # construct gaussian filter
        gFilter = np.zeros((winSize, winSize))
        # translate, (offX, offY)
        offVector = (int(winSize/2), int(winSize/2))

        scale = 1/(2*np.pi*sigma**2)
        precision = 0
        for yIndex in range(0, winSize):
            y = yIndex - offVector[1]
            for xIndex in range(0, winSize):
                x = xIndex - offVector[0]
                temp = -(x**2+y**2)/(2*sigma**2)
                gFilter[yIndex][xIndex] = scale*np.exp(temp)
                precision += gFilter[yIndex][xIndex]
        print('The Gaussian Filter is: \n')
        print(gFilter)
        print('\n')
        print('The precision is: %s \n'%precision)
        # filtering target image.
        height, width = befIm.shape[:2]
        tmpIm = np.zeros((height, width), dtype='float32')
        print('Gaussian Filtering Start!')
        # y, x - loop for every pixel in image
        for y in range(0, height):
            for x in range(0, width):
                # yWin, xWin - loop for pixels in filter window
                for yWin in range(-offVector[1], offVector[1]+1):
                    for xWin in range(-offVector[0], offVector[0]+1):
                        tempIntensity = 0
                        xImInd = x + xWin
                        yImInd = y + yWin
                        xFilInd = xWin + offVector[0]
                        yFilInd = yWin + offVector[1]
                        # 8 conditions??? Any better method?
                        # left side, no corners
                        if xImInd < 0 and yImInd >= 0 and yImInd < height:
                            tempIntensity = befIm[yImInd][0]
                        # right side, no corners
                        elif xImInd >= width and yImInd >=0 and yImInd < height:
                            tempIntensity = befIm[yImInd][width-1]
                        # up side, no corners
                        elif yImInd < 0 and xImInd >=0 and xImInd < width:
                            tempIntensity = befIm[0][xImInd]
                        # bottom side, no corners
                        elif yImInd >= height and xImInd >=0 and xImInd < width:
                            tempIntensity = befIm[height-1][xImInd]
                        # up-left corner
                        elif xImInd < 0 and yImInd < 0:
                            tempIntensity = befIm[0][0]
                        # up-right corner
                        elif xImInd >= width and yImInd < 0:
                            tempIntensity = befIm[0][width-1]
                        # bottom-left corner
                        elif xImInd < 0 and yImInd >= height:
                            tempIntensity = befIm[height-1][0]
                        # bottom-right corner
                        elif xImInd >= width and yImInd >= height:
                            tempIntensity = befIm[height-1][width-1]
                        # inside of befor image
                        else:
                            tempIntensity = befIm[yImInd][xImInd]
                        tmpIm[y][x] += tempIntensity * gFilter[yFilInd][xFilInd]
                        if tmpIm[y][x] > 255:
                            tmpIm[y][x] = 255
        print(tmpIm)
        aftIm = tmpIm.astype(int)
        print(aftIm)
        return aftIm




    def sobel_filtering(self, befIm, axis):
        print(befIm)
        if axis == 0:
            sFilter = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        elif axis == 1:
            sFilter = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        height, width = befIm.shape[:2]
        aftIm = np.zeros((height, width), dtype='int16')
        for y in range(0, height):
            for x in range(0, width):
                # yWin, xWin - loop for pixels in filter window
                for yWin in range(-1, 2):
                    for xWin in range(-1, 2):
                        tempIntensity = 0
                        xImInd = x + xWin
                        yImInd = y + yWin
                        xFilInd = xWin + 1
                        yFilInd = yWin + 1
                        # 8 conditions??? Any better method?
                        # left side, no corners
                        if xImInd < 0 and yImInd >= 0 and yImInd < height:
                            tempIntensity = befIm[yImInd][0]
                        # right side, no corners
                        elif xImInd >= width and yImInd >=0 and yImInd < height:
                            tempIntensity = befIm[yImInd][width-1]
                        # up side, no corners
                        elif yImInd < 0 and xImInd >=0 and xImInd < width:
                            tempIntensity = befIm[0][xImInd]
                        # bottom side, no corners
                        elif yImInd >= height and xImInd >=0 and xImInd < width:
                            tempIntensity = befIm[height-1][xImInd]
                        # up-left corner
                        elif xImInd < 0 and yImInd < 0:
                            tempIntensity = befIm[0][0]
                        # up-right corner
                        elif xImInd >= width and yImInd < 0:
                            tempIntensity = befIm[0][width-1]
                        # bottom-left corner
                        elif xImInd < 0 and yImInd >= height:
                            tempIntensity = befIm[height-1][0]
                        # bottom-right corner
                        elif xImInd >= width and yImInd >= height:
                            tempIntensity = befIm[height-1][width-1]
                        # inside of befor image
                        else:
                            tempIntensity = befIm[yImInd][xImInd]
                        aftIm[y][x] += tempIntensity * sFilter[yFilInd][xFilInd]
                if aftIm[y][x] < 0:
                    aftIm[y][x] = -aftIm[y][x]
                if aftIm[y][x] > 255:
                    aftIm[y][x] = 255
        print(aftIm)
        return aftIm

    def sobel_combining(self, hIm, vIm, threshold):
        im = hIm + vIm
        height, width = im.shape[:2]
        print(height, width)
        for y in range(0, height):
            for x in range(0, width):
                if im[y][x] > 255:
                    im[y][x] = 255
                if im[y][x] < threshold:
                    im[y][x] = 0
        return im

    def non_max_suppression(self, befIm):
        height, width = befIm.shape[:2]
        aftIm = np.array(befIm, copy=True)
        for y in range(0, height):
            for x in range(0, width):
                xDeriv = 0
                yDeriv = 0
                theta = 0
                if x == width-1 and y < height-1:
                    xDeriv = 0
                    yDeriv = 1
                elif x < width-1 and y == height-1:
                    xDeriv = 1
                    yDeriv = 0
                elif x == width-1 and y == height-1:
                    xDeriv = 0
                    yDeriv = 0
                else:
                    xDeriv = befIm[y][x+1] - befIm[y][x]
                    yDeriv = befIm[y+1][x] - befIm[y][x]
                # cal theta
                if xDeriv == 0 and yDeriv == 0:
                    #xDeriv = 0
                    aftIm[y][x] = 0
                elif xDeriv == 0 and yDeriv != 0:
                    theta = np.pi / 2 #same to -pi/2
                else:
                    theta = np.arctan(yDeriv/xDeriv)

                # classify theta
                # 1, vertical
                if (theta >= -np.pi/2 and theta < -3*np.pi/8) or (theta >= 3*np.pi/8 and theta < np.pi/2):
                    if y > 0 and y < height-1:
                        if befIm[y][x] < befIm[y+1][x] or befIm[y][x] < befIm[y-1][x]:
                            aftIm[y][x] = 0
                    if y == 0 and befIm[y][x] < befIm[y+1][x]:
                        aftIm[y][x] = 0
                    if y == height-1 and befIm[y][x] < befIm[y-1][x]:
                        aftIm[y][x] = 0
                # 2, horizontal
                if theta >= -np.pi/8 and theta < np.pi/8:
                    if x > 0 and x < width-1:
                        if befIm[y][x] < befIm[y][x+1] or befIm[y][x] < befIm[y][x-1]:
                            aftIm[y][x] = 0
                    if x == 0 and befIm[y][x] < befIm[y][x+1]:
                        aftIm[y][x] = 0
                    if x == width-1 and befIm[y][x] < befIm[y][x-1]:
                        aftIm[y][x] = 0
                # 3, D1, RU
                if theta >= -3*np.pi/8 and theta < -np.pi/8:
                    #left side, up side and up-left corner
                    if x > 0 and y > 0:
                        if befIm[y][x] < befIm[y-1][x+1] or befIm[y][x] < befIm[y+1][x-1]:
                            aftIm[y][x] = 0
                    if x == 0 and y != 0:
                        if befIm[y][x] < befIm[y-1][x+1]:
                            aftIm[y][x] = 0
                    if x != 0 and y == 0:
                        if befIm[y][x] < befIm[y+1][x-1]:
                            aftIm[y][x] = 0
                # 4, D2, LU
                if theta >= np.pi/8 and theta < 3*np.pi/8:
                    #left side, up side and up-left corner
                    if x > 0 and y > 0:
                        if befIm[y][x] < befIm[y-1][x-1] or befIm[y][x] < befIm[y+1][x+1]:
                            aftIm[y][x] = 0
                    if (x == 0 and y != 0) or (x != 0 and y == 0):
                        if befIm[y][x] < befIm[y+1][x+1]:
                            aftIm[y][x] = 0
        return aftIm



if __name__ == '__main__':
    #1, read pgm to ndarray
    im0 = np.array(Image.open('./cs558s16_hw1/kangaroo.pgm'), dtype='int16')
    im1 = np.array(Image.open('./cs558s16_hw1/plane.pgm'), dtype='int16')
    im2 = np.array(Image.open('./cs558s16_hw1/red.pgm'), dtype='int16')

    #(2), save to png
#    tarIm0 = scipy.misc.imsave('./images/kangaroo.png', im0)
#    tarIm1 = scipy.misc.imsave('./images/plane.png', im1)
#    tarIm2 = scipy.misc.imsave('./images/red.png', im2)

    #2, Create Filter instance and filtering images
    f = Filters()
    # do Gaussian filtering, user specified sigma
    img0_gaussian = f.gaussian_filtering(im1,1)
    img0_sobel_h = f.sobel_filtering(img0_gaussian,0)
    img0_sobel_v = f.sobel_filtering(img0_gaussian,1)
    img0_sobel = f.sobel_combining(img0_sobel_h, img0_sobel_v, 100)
    img0_nms = f.non_max_suppression(img0_sobel)
    tarIm0_gaussian = scipy.misc.imsave('./images/kangaroo_gaussian.png', img0_gaussian)
    tarIm0_sobel = scipy.misc.imsave('./images/kangaroo_sobel.png', img0_sobel)
    tarIm0_nms = scipy.misc.imsave('./images/kangaroo_nms.png', img0_nms)
    #testG = np.array(Image.open('./images/kangaroo_gaussian.png'), dtype='int16')
    #testSV = f.sobel_filtering(testG, 0)
    #testSH = f.sobel_filtering(testG, 1)
    #testS = f.sobel_combining(testSV, testSH, 120)
    #test = f.non_max_suppression(testS)
    #scipy.misc.imsave('./images/test_nm.png', test)

    #sys0_gaussian = ndimage.filters.gaussian_filter(im0,5)
    #tarSys0_gaussian = scipy.misc.imsave('./images/kangaroo_sys.png', sys0_gaussian)
