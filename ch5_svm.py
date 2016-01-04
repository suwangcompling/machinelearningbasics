"""
Author: Su Wang
Institution: Linguistics Department, University of Texas at Austin
Model: Simple SVM
"""

'''
The cost function of SVM as introduced in the tutorial is
similar to that of logistic regression, and therefore
omitted here to avoid repetition.
The implementation of it requires only some slight changes
to the logistic regression version.
The SVM cost functions are in (5.8), (5.9).
'''

import numpy as np

# gaussian kernel
def gaussian_kernel(x1, x2, sigma):
	return np.exp(-sum((x1 - x2) **2.0) / (2*sigma**2.0))

