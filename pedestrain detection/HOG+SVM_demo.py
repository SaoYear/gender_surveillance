
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pickle
import time


class PedestrianDetection(object):
    def __init__(self, img_dir='C:/Users/sn_96/Desktop/FYP_Gender_Processing/Database/pd_dataset/demo/pd_2.jpg'):
        self.img_dir = img_dir
        self.img = self.get_data(self.img_dir)     # Image list, address
        # print(self.img.shape)
        self.img = cv.resize(self.img, (500, 300))
        # print(self.img.shape)
        self.hog = self.def_hog()
        self.clf = self.get_clf()
        self.detection()
        cv.destroyAllWindows()

    @staticmethod
    def get_data(img_dir):
        return cv.imread(img_dir)

    @staticmethod
    def get_clf():
        # print("Loading the model", end='')
        with open('model.pickle', 'rb') as f:
            _clf = pickle.load(f)
        # print(" ...")
        return _clf

    def detection(self):
        # print("Detecting the pedestrians ...")
        x_axis = self.img.shape[0]
        # print(x_axis)
        y_axis = self.img.shape[1]
        end_point = (x_axis - 160, y_axis - 96)
        # print(end_point)
        img_gray = cv.cvtColor(self.img, cv.COLOR_RGB2GRAY)
        # print("Prediction:")
        for x in range(end_point[0])[::80]:
            for y in range(end_point[1])[::48]:
                img_window = img_gray[x: x+160, y: y+96]
                if img_window.shape != (160, 96):
                    pass
                # print("Warning! the picture is wrong!")
                window_hog = np.reshape(self.hog.compute(img_window), (1, 7524))
                prediction = self.clf.predict(window_hog)
                # cv.imshow("test", img_window)
                # cv.waitKey()
                # print(prediction, end='')
                if prediction == 1:
                    # print(x, y)
                    cv.rectangle(self.img, (y, x), (y+96, x+160), (0, 255, 0), 2)
        # print("\nThe result of PD has been shown.")
        cv.namedWindow('PD result', cv.WINDOW_AUTOSIZE)
        cv.imshow("PD result", self.img)
        cv.waitKey()

    def def_hog(self):
        _feature = 0
        # print("Initializing HOG descriptor...")
        winsize = (96, 160)
        # print(" window size:", winsize)
        blocksize = (16, 16)
        blockstride = (8, 8)
        cellsize = (8, 8)
        nbins = 9
        hog = cv.HOGDescriptor(winsize,
                               blocksize,
                               blockstride,
                               cellsize,
                               nbins)
        return hog


start = time.time()
step_1 = PedestrianDetection()
stop = time.time()
print("Total time:", stop - start)
