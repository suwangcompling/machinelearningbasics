"""
Author: Su Wang
Institution & Copyright: Linguistics Department, University of Texas at Austin
Model: Linear Regression
Description: Code written in verbose fashion for pedagogical reason
"""

import numpy as np

class Lin_Reg(object):

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

    # cost function
    def cost(self):
        prediction = np.dot(self.X,self.theta)
        J = (1.0/(2*self.m)) * sum((prediction-self.y)**2)
        return J

    # gradient descent
    def grad_desc(self, alpha, iterations): # alpha: learning rate
        for i in range(iterations):
            prediction = np.dot(self.X,self.theta)
            delta = prediction-self.y
            for j in range(self.n):
                self.theta[j] = self.theta[j] - \
                                alpha*(1.0/self.m)*sum(delta*self.X[:,j].reshape(self.m,1))
        return

    # current weights
    def get_weights(self):
        return self.theta

'''
Usage Example
File: comma separated house_price.txt  
area  num_bedrooms  price  
2104,3,399900  
1600,3,329900  
2400,3,369000  
1416,2,232000  
3000,4,539900  
1985,4,299900  
...  
'''
>>> lr = lin_reg.Lin_Reg('house_price.txt')  
>>> lr.cost()
array([ 32.07273388])
>>> lr.get_weights()
array([[ 0.],
       [ 0.]])
>>> lr.grad_desc(0.01,1500)
>>> lr.get_weights()
array([[-3.63029144],
       [ 1.16636235]]
>>> lr.cost()
array([ 4.48338826])


