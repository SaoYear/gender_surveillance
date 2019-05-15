# READ FILE NAME USE OS
# import os
#
# filename = []
# root = []
# temp = []
# for r, t, file in os.walk('DIR/PETA dataset_19000/'):
#     filename.append(file)
#     root.append(r)
#     temp.append(t)
# print(filename)
# print(root)
# print(temp)

import cv2 as cv
img = cv.imread('DIR/pd_1.jpg')
print(img.shape)
cv.namedWindow('pic1', cv.WINDOW_AUTOSIZE)
cv.imshow("pic1", img[5: 5+160, 5: 120])
cv.waitKey()
