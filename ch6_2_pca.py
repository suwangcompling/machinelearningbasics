"""
Author: Su Wang
Institution: Linguistics Department, University of Texas at Austin
Model: Principal Component Analysis (PCA) # a demo for 2D data
"""

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from matplotlib import lines

class PCA:

    # member variables
    X, Xnorm, Xproj, Xrec = None, None, None, None # data
    Sigma = None # covariance matrix
    U, S, V = None, None, None # svd matrices
    k = None # pca dimension

    # constructor
    def __init__(self, filename, k):
        dm = scio.loadmat(filename)
        self.k = k
        self.X = dm['X']
        self.normalize()
        self.cov_matrix()
        self.svd()
        self.project()
        self.recover()

    # normalize data
    def normalize(self):
        self.Xnorm = np.apply_along_axis(lambda x: (x - np.mean(x)) / np.std(x), 0, self.X)
        return

    # compute covariance matrix
    def cov_matrix(self):
        self.Sigma = (1.0/self.Xnorm.shape[0]) * np.dot(self.Xnorm.T,self.Xnorm)
        return

    # singular value decomposition
    def svd(self):
        self.U, self.S, self.V = np.linalg.svd(self.Sigma)
        return

    # PC ''measurement'': project data onto PCs
    def project(self):
        self.Xproj = np.dot(self.Xnorm, self.U[:,0:self.k])
        return

    # recover data: approximation
    def recover(self):
        self.Xrec = np.dot(self.Xproj, self.U[:,0:self.k].T)
        return

    # demo: projection
    def projection_demo(self):
        plt.plot(self.Xnorm[:,0],self.Xnorm[:,self.k],'ro')
        plt.plot(self.Xrec[:,0],self.Xrec[:,self.k],'bo',color='green')
        for i in range(0,self.Xrec.shape[0]):
            plt.gca().add_line(lines.Line2D(xdata=[self.Xnorm[i,0],self.Xrec[i,0]],\
                                            ydata=[self.Xnorm[i,1],self.Xrec[i,1]],\
                                            c='g',lw=1,ls='--'))
        return

'''
Usage Example
pca = PCA('2d_data.mat', 1)
pca.projection_demo()
'''
