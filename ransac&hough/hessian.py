import numpy as np
import scipy
from scipy import ndimage
from PIL import Image

def gaussian_filtering(befIm, sigma=1):
    # use 3sigma principle
    winSize = sigma*6 + 1
    # construct gaussian filter
    gFilter = np.zeros((winSize, winSize))
    # translate, (offX, offY)
    offVector = (int(winSize/2), int(winSize/2))

    scale = 1/(2*np.pi*sigma**2)
    sumOfWeight = 0
    for yIndex in range(0, winSize):
        y = yIndex - offVector[1]
        for xIndex in range(0, winSize):
            x = xIndex - offVector[0]
            temp = -(x**2+y**2)/(2*sigma**2)
            gFilter[yIndex][xIndex] = scale*np.exp(temp)
            sumOfWeight += gFilter[yIndex][xIndex]
    # normalize sumOfWeight
    gFilter = gFilter/sumOfWeight

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
    aftIm = tmpIm.astype(int)
    print('Gaussian Filtering ends.')
    return aftIm

def normalizing(befIm_2d):
    height, width = befIm_2d.shape[:2]
    befIm_1d = befIm_2d.flatten()
    aftIm_1d = np.sort(befIm_1d)
    aftIm_2d = np.zeros((height, width), dtype='int32')
    min = aftIm_1d[int(height*width*0.01)]
    max = aftIm_1d[int(height*width*0.99)]
    print(min)
    print(max)
    for y in range(0, height):
        for x in range(0, width):
            if befIm_2d[y][x] < min:
                befIm_2d[y][x] = 0
            elif befIm_2d[y][x] > max:
                befIm_2d[y][x] = 0
            else:
                aftIm_2d[y][x] = int((befIm_2d[y][x]-min)/(max-min)*255)
            if aftIm_2d[y][x] > 255 or aftIm_2d[y][x] < 0:
                print('Error: Out of boundary.')
    return aftIm_2d

def sobel_filtering(befIm, axis):
    if axis == 0:
        sFilter = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    elif axis == 1:
        sFilter = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    height, width = befIm.shape[:2]
    aftIm = np.zeros((height, width), dtype='int32')
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
    return aftIm

def non_max_suppression(befImg):
    h, w = befImg.shape[:2]
    #aftImg = np.zeros((h,w), dtype='int32')
    aftImg = np.copy(befImg)
    rest = 0
    for y in range(0, h):
        for x in range(0, w):
            isLocalMax = True
            for j in range(y-1, y+2):
                for i in range(x-1, x+2):
                    if i >= 0 and i < w and j >= 0 and j < h:
                        if befImg[j][i] >= befImg[y][x] and (j != y or i != x):
                            aftImg[y][x] = 0
                            isLocalMax = False
            if isLocalMax:
                rest += 1
    print('There are %d points left.'%rest)
    return aftImg

def cutoff(befImg, threshold=160):
    height, width = befImg.shape[:2]
    for j in range(0, height):
        for i in range(0, width):
            if befImg[j][i] < threshold:
                befImg[j][i] = 0
    return befImg


if __name__ == '__main__':
    im = np.array(Image.open('./road.png'))
    tempImg = gaussian_filtering(im)
    Ix = sobel_filtering(tempImg, 0)
    Iy = sobel_filtering(tempImg, 1)
    Ixx = sobel_filtering(Ix, 0)
    scipy.misc.imsave('./Ixx.png', Ixx)
    Iyy = sobel_filtering(Iy, 1)
    scipy.misc.imsave('./Iyy.png', Iyy)
    Ixy = sobel_filtering(Ix, 1)
    scipy.misc.imsave('./Ixy.png', Ixy)

    afterSobel = normalizing(Ixx*Iyy-Ixy*Ixy)
    scipy.misc.imsave('./normalization.png', afterSobel)
    #cutoff(afterSobel)
    afterCutoff = cutoff(afterSobel)
    scipy.misc.imsave('./cutoff.png', afterCutoff)
    afterNMS = non_max_suppression(afterCutoff)
    scipy.misc.imsave('./nms.png', afterNMS)
    #result3 = cutoff(result2)
    #scipy.misc.imsave('./hessian2.png', result3)

   # testIm = np.array(Image.open('./bwg.gif'))
   # tIx = sobel_filtering(testIm, 0)
   # tIy = sobel_filtering(testIm, 1)
   # tIxx = sobel_filtering(tIx, 0)
   # tIyy = sobel_filtering(tIy, 1)
   # tIxy = sobel_filtering(tIx, 1)
   # tres = normalizing(tIxx*tIyy-tIxy*tIxy)
   # r = non_max_suppression(tres)
   # scipy.misc.imsave('./tres.png', r)
