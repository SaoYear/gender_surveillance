import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
start = time.time()
# 计算hog特征
def get_hog(image):
    winSize = (image.shape[1], image.shape[0])
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9

    hog = cv.HOGDescriptor(winSize,
                           blockSize,
                           blockStride,
                           cellSize,
                           nbins)
    hist = hog.compute(image)
    return hist

src = cv.imread('DIR/000_1_1_.bmp')
plt.imshow(src, cmap='gray_r')

plt.show()
hog = get_hog(src)
cv.waitKey(15)
stop = time.time()
print("time:", (stop - start) * 1000, 'ms')
print(np.shape(hog))

cv.destroyAllWindows()
