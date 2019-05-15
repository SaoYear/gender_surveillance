from sklearn.utils import check_random_state
from sklearn import svm, datasets
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import random
import pickle
import time


class PedestrianDetection(object):
    def __init__(self,  data_dir='C:/Users/sn_96/Desktop/FYP_Gender_Processing/Database/pd_dataset/'):
        self.data_dir = data_dir
        self.data, self.img_dir = self.get_data()     # Image list, address
        print('The size of the images:', end='')
        print(self.data[0].shape[0], '×', end='')
        print(self.data[0].shape[1])
        self.number_example = len(self.data)
        self.label = self.get_label()  # Get label, pos - 1, neg - 0
        self.feature = []
        self.get_hog()        # HOG Feature matrix
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_data()
        self.clf = self.SVM_clf()
        self.test()

    def get_data(self):
        print("Reading the data...")
        pos = []    # Positive and negative samples
        neg = []
        for _, _, file in os.walk(self.data_dir + 'pos/'):
            pos = file
        pos = [self.data_dir + 'pos/' + item for item in pos]
        print(" Positive samples complete!")
        for _, _, file in os.walk(self.data_dir + 'neg/'):
            neg = file
        neg = [self.data_dir + 'neg/' + item for item in neg]
        print(" Negative samples complete!")
        _data = pos + neg
        print(" Shuffling...")
        random.shuffle(_data)
        _imgs = [cv.imread(item, 0) for item in _data]
        print("Complete!")
        return _imgs, _data

    def get_label(self):
        print("Extracting the label of the data...")
        _dir = [item.split('/')[-1].split('_')[0] for item in self.img_dir]
        _label = [1 if item == 'pos' else 0 for item in _dir]       # pos: 1 neg: 0
        print(" Label:", _label)
        print("Complete!")
        return _label

    def get_hog(self):
        print("Initializing HOG descriptor...")
        winsize = (self.data[0].shape[1], self.data[0].shape[0])
        print(" window size:", winsize)
        blocksize = (16, 16)
        blockstride = (8, 8)
        cellsize = (8, 8)
        nbins = 9
        hog = cv.HOGDescriptor(winsize,
                               blocksize,
                               blockstride,
                               cellsize,
                               nbins)
        print(hog.checkDetectorSize())
        print("Getting the HOG features of the images")
        for id, item in enumerate(self.data):
            if item.shape != (160, 96):
                print("Warning! the picture of {} is wrong!".format(id))
                print(self.img_dir[id])
            if id % (self.number_example // 7) == 0:
                print('.', end='')
            hist = np.reshape(hog.compute(item), hog.compute(item).size)   # size: 7524 * 1
            self.feature.append(hist)
        print("\nComplete!")

    def split_data(self):   # split data into train and test for SVM
        print("Split the data in to train and test")
        _x_train, _x_test, _y_train, _y_test = ms.train_test_split(self.feature,
                                                                   self.label,
                                                                   random_state=1,
                                                                   train_size=0.8)
        print("Complete!")
        return _x_train, _x_test, _y_train, _y_test

    def SVM_clf(self, kernel='rbf'):
        print("Training the SVM classifier...")
        clf = svm.SVC(kernel=kernel).fit(self.x_train, self.y_train)
        with open('model.pickle', 'wb') as f:
            pickle.dump(clf, f)
        print("Complete!")
        return clf

    def test(self):
        print("Testing the SVM classifier")
        print("Accuracy on test dataset: %.5f" % (self.clf.score(self.x_test, self.y_test)))


step_1 = PedestrianDetection()
