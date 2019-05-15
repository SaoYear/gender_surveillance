from skimage.feature import hog
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import time

im = io.imread('C:/Users/sn_96/Desktop/FYP_Gender_Processing/Database/data_test/000_1_1_.bmp')
start = time.process_time()


im_gy = io.imread('C:/Users/sn_96/Desktop/FYP_Gender_Processing/Database/data_test/000_1_1_.bmp', as_gray=True)
normalised_blocks, hog_image = hog(im_gy, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
stop = time.process_time()
hot_image = np.array(hog_image, dtype=float)
normalised_blocks = np.array(normalised_blocks, dtype=float)

print(np.shape(normalised_blocks))
print("time:", (stop - start) * 1000, 'ms')
plt.subplot(121)
plt.imshow(im, cmap='gray')
# plt.subplot(132)
# plt.imshow(im_gy, cmap='gray')
plt.subplot(122)
plt.imshow(hot_image, cmap='gray')
plt.show()

