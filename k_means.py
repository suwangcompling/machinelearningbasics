"""
Author: Su Wang
Institution: Linguistics Department, University of Texas at Austin
Model: K-means
"""

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import random
import scipy.spatial.distance as scsd

class KMEANS(object):

    # member variables
    X = None # data
    dl, cl = None, None # data list; class landmarks

    # constructor
    def __init__(self, filename):
        dm = scio.loadmat(filename)
        self.X = dm['X']
        # convert data into a list tuples to allow labeling
        # data points under class landmarks
        self.dl = []
        for i in range(self.X.shape[0]):
            self.dl.append([0,self.X[i]])

    # plot data for deciding k (only for 2D data set)
    def plot_data(self):
        x1 = self.X.T[0]
        x2 = self.X.T[1]
        plt.plot(x1,x2,'ro')

    # deciding on k
    # randomly sample k ''coordinates'' for class landmarks
    def sample_class_landmarks(self, k):
        self.cl = random.sample(self.X,k)

    # function: compute distance between data points and class landmarks
    #   and label data points under closest class landmark
    def find_closest(self):
        for item in self.dl:
            minDistIndex = np.argmin([scsd.euclidean(item[1],self.cl[j]) for j in range(len(self.cl))])
            item[0] = minDistIndex

    # function: set class landmarks to centroids
    def reset_class_landmarks(self):
        counter = [[] for i in range(len(self.cl))]
        offset = True
        for item in self.dl:
            counter[item[0]].append(item[1])
        for i in range(len(counter)):
            if np.all(self.cl[i] == sum(counter[i])/len(counter[i])):
                offset = False
            self.cl[i] = sum(counter[i])/len(counter[i])
        return offset

    # function: k-means algorithm
    def k_means(self, printIterations=False):
        offset = True
        iterations = 0
        while offset == True:
            iterations += 1
            self.find_closest()
            offset = self.reset_class_landmarks()
        if printIterations == True:
            print '%s iterations to reach (local) minima.' % iterations
        return self.cl

    # show class landmarks (only for 2D)
    def show_class_landmarks(self):
        x = [item[0] for item in self.cl]
        y = [item[1] for item in self.cl]
        plt.plot(x,y,'ro',color='green',markersize=10)

'''
Usage Example
File: 2D data set 2d_example.mat
km = KMEANS('2d_example.mat')
km.plot_data()
km.sample_class_landmarks(3)
km.k_means(True)
km.show_class_landmarks()
'''
