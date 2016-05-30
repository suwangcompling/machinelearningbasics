"""
Author: Su Wang
Institution: Linguistics Department, University of Texas at Austin
Model: Neural Network
"""

import numpy as np
import scipy.io as scio
import scipy.optimize as sco

class NN: # 3-layered neural network

    # member variables
    dm = None # data matrix: last column is dep. vars
    X, y = None, None # indep. vars, dep. vars
    m, n = 0, 0 # training size, var. size
    Theta1, Theta2 = None, None # weights (neural network with 1 hidden layer)
    a1, z2, a2, z3, a3 = None, None, None, None, None # for fwd & bwd propagation
    vecDim, yVec = None, None # for vectorization of y
    T1Flattened, T2Flattened = None, None # unrolled parameters, for optimization

    # constructor
    def __init__(self, filename): # data & weights will be stored in .mat files
        self.dm = scio.loadmat(filename)
        self.X = self.dm['X']
        self.y = self.dm['y']
        self.m, self.n = self.X.shape

    # sigmoid function
    def sigmoid(self, prediction):
        return 1.0/(1+np.exp(-prediction))

    # weights initialization (values in [-.12,.12) )
    def wt_init(self, T1Shape, T2Shape): # T1/2Shape are 2-tuples indicating dimensions of weight matrices
        self.Theta1 = (np.random.random(T1Shape)*.24)-.12
        self.Theta2 = (np.random.random(T2Shape)*.24)-.12
        self.param_flatten() # initialize flat/1D data container too

    # forward propagation
    def fwd_prop(self, printAccuracy=False):
        # propagation step
        self.a1 = np.concatenate( (np.ones((1,self.m)),self.X.T), axis=0)
        self.z2 = np.dot(self.Theta1,self.a1)
        self.a2 = np.concatenate( (np.ones((1,self.m)),self.sigmoid(self.z2)), axis=0)
        self.z3 = np.dot(self.Theta2,self.a2)
        self.a3 = self.sigmoid(self.z3) 
        if printAccuracy == True:
            # prediction & accuracy evaluation step
            result = self.a3.T
            prediction = np.zeros((self.m,1))
            for i in range(result.shape[0]): 
                prediction[i] = np.argmax(result[i])+1 # +1 to convert from 0-index to 1-index
            # compute accuracy (percentage of correct predictions)
            # np.where gives 1 to where prediction gets it right, 0 otherwise
            # so the sum of this is the number of correct predictions
            return sum(np.where((self.y-prediction)==0,1,0))/float(self.m)
        else:
            return

    # vectorization
    # (y is usually given with, e.g. 1-10 class label, however we'd like it to
    # be represented in vectors. e.g. [0,0,1] for 3)
    def vectorization(self):
        vec = np.zeros((len(self.y),self.a3.shape[0])) 
        for i in range(len(self.y)):
            vec[i][self.y[i]-1] = 1 # turning class labels to vector representation
        self.yVec = vec.T # such that the vectorized y is of the same shape as a3 
        return

    # flatten & unflatten parameters
    # a. convenience for flattening class vars.
    def param_flatten(self):
        self.T1Flattened = self.Theta1.T.reshape(1,self.Theta1.size)
        self.T2Flattened = self.Theta2.T.reshape(1,self.Theta2.size)
    def param_unflatten(self):
        self.Theta1 = self.T1Flattened.T.reshape(self.Theta1.T.shape).T
        self.Theta2 = self.T2Flattened.T.reshape(self.Theta2.T.shape).T
    # b. generic flattening/unflattening
    def matrix_flatten(self, mtrx):
        return mtrx.T.reshape(1,mtrx.size)

    # cost function
    def cost(self, Theta, lmd): # lmd: lambda, regularization coefficient; Theta: flattened param/weights input
        # Unflatten parameters
        self.T1Flattened, self.T2Flattened = Theta[0:self.Theta1.size], Theta[self.Theta1.size:]
        self.param_unflatten() # "convert" the input weights into matrix-form and store locally
        # forward propagation step
        self.fwd_prop() # a1,2,3; z2,3 are computed
        # vectorizing y
        self.vectorization()
        # compute cost
        regularization = ((self.Theta1[:,1:]**2).sum() + (self.Theta2[:,1:]**2).sum()) * \
                            lmd/(2.0*self.m) # do not regularize bias by convention
        J = (1.0/self.m)*sum(sum(-self.yVec*np.log(self.a3) - \
                                 (1.0-self.yVec)*np.log(1.0-self.a3))) + \
                                regularization
        return J

    # backward propagation
    def gradient(self, Theta, lmd):
        # Unflatten parameters
        self.T1Flattened, self.T2Flattened = Theta[0:self.Theta1.size], Theta[self.Theta1.size:]
        self.param_unflatten() # "convert" the input weights into matrix-form and store locally
        # vectorizing y
        self.vectorization()
        # initializing error accumulators
        Delta1 = np.zeros((self.Theta1.shape))
        Delta2 = np.zeros((self.Theta2.shape))
        # learning loop
        for i in range(self.m):
            # forward propagation step
            xi = self.X[i].reshape(len(self.X[i]),1)
            a1 = np.concatenate( (np.ones((1,1)),xi), axis=0)
            z2 = np.dot(self.Theta1,a1)
            a2 = np.concatenate( (np.ones((1,1)),self.sigmoid(z2)), axis=0)
            z3 = np.dot(self.Theta2,a2)
            a3 = self.sigmoid(z3)
            # backward propagation step
            delta3 = self.a3 - self.yVec[:,i].reshape(len(self.yVec[:,i]),1) 
            delta2 = np.dot((self.Theta2.T[1:]),delta3) * self.sigmoid(z2)
            Delta1 = Delta1 + np.dot(delta2,self.a1.T)
            Delta2 = Delta2 + np.dot(delta3,self.a2.T)
            Theta1Grad = (1.0/self.m)*(Delta1 + lmd*np.concatenate( (np.zeros((self.Theta1.shape[0],1)),self.Theta1[:,1:]), axis=1))
            Theta2Grad = (1.0/self.m)*(Delta2 + lmd*np.concatenate( (np.zeros((self.Theta2.shape[0],1)),self.Theta2[:,1:]), axis=1))
        return np.append(self.matrix_flatten(Theta1Grad), self.matrix_flatten(Theta2Grad))

    # fmincg optimization (REAL SLOW ON LARGE DATA SET, UNDER IMPROVEMENT!!)
    def fmincg(self, Theta, lmd):
        result = sco.fmin_cg(self.cost, fprime=self.gradient, x0=Theta, args=(lmd), \
                           maxiter=50, full_output=True)
        print result[1]



'''
Usage Example
file: 400 by 400 pixels handwritten digits
nn = NN('handwritten_digit.mat')
nn.wt_init((25,401),(10,26))
Theta = np.append(nn.T1Flattened,nn.T2Flattened)
lmd=1.0
nn.cost(Theta,lmd)
6.5290172972694931
nn.gradient(Theta,lmd) # produces two weight matrices in one 1D array for fmincg computation. can be reconstructed.
nn.fmincg(Theta,(lmd,)) # args input must be in tuple. REAL SLOW ON LARGE DATA SET, UNDER IMPROVEMENT!!
'''

