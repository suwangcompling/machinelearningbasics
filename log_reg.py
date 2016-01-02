"""
Author: Su Wang
Institution & Copyright: Linguistics Department, University of Texas at Austin
Model: Logistic Regression
"""

import numpy as np
import scipy.optimize as sco

class Log_Reg(object):

    # member variables
    dm = None # data matrix: last column is dep. vars
    X, y = None, None # indep. vars, dep. vars
    m, n = 0, 0 # training size, var. size
    theta = None # weights

    # constructor
    def __init__(self, filename):
        self.dm = np.loadtxt(filename, delimiter=',')
        self.m, self.n = self.dm.shape # count number of rows
        self.X = np.concatenate((np.ones((self.m,1)),self.dm[:,0:-1]),axis=1) # including bias
        self.y = self.dm[:,-1].reshape(self.m,1)
        self.theta = np.zeros((self.n,1)) # one weight for bias

    # sigmoid function
    def sigmoid(self, prediction):
        return 1.0/(1+np.exp(-prediction))

    # cost function
    def cost(self,theta):
        prediction = np.dot(self.X,theta)
        J = (1.0/self.m) * (np.dot(np.log(self.sigmoid(prediction)).T,-self.y) - \
                       np.dot(np.log(1.0-self.sigmoid(prediction)).T,(1-self.y)))
        return J

    # fminunc optimization
    def fminunc(self):
        result = sco.fmin(self.cost,x0=self.theta,maxiter=500,full_output=True)
        print 'optimized weights: %s' % result[0]
        print 'optimized cost: %s' % result[1]
        return

'''
Usage Example
File: comma separated admission.txt
34.62365962451697,78.0246928153624,0
30.28671076822607,43.89499752400101,0
35.84740876993872,72.90219802708364,0
60.18259938620976,86.30855209546826,1
79.0327360507101,75.3443764369103,1
45.08327747668339,56.3163717815305,0
...
'''
>>> lr = Log_Reg('admission.txt')
>>> lr.cost(lr.theta)
array([[ 0.69314718]])
>>> lr.fminunc()
Optimization terminated successfully.
         Current function value: 0.203498
         Iterations: 157
         Function evaluations: 287
optimized weights: [-25.16130062   0.20623142   0.20147143]
optimized cost: 0.20349770159
