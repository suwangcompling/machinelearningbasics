### DIMENSIONALITY REDUCTION


## PCA: Mechanism

# basic calculation: covariance
import numpy as np
X = [[2.,0.,-1.4],
     [2.2,.2,-1.5],
     [2.4,.1,-1.],
     [1.9,0.,-1.2]]
    # each col is a 4-dimensional variable.
    # use .T transpose to compute correctly,
    # because a variable is supposed to look like:
    # x_1 = [2.,2.2,2.4,1.9]
cov_X = np.cov(np.array(X))

# basic calculation: eigenvectors/eigenvalues
#   direction & magnitude of a vector.
#   i) eigenvector & eigenvalue
#       **A PC is an eigenvector of its covariance matrix**
#       if the corresponding eigenvalue is the greatest, then
#       the eigenvector is the 1st PC.
#   ii) normalization
#       PCA requires unit eigenvectors. an eigenvector can be
#       normalized by dividing it by its norm:
#       ||x|| = \sqrt(x_1^2 + ... + x_p^2)
import numpy as np
w, v = np.linalg.eig(np.array([[.8,.3],[.2,.7]]))
    # this represents matrix:
    # 1 -2
    # 2 -3
w
    # lmd1 lmd2
    # [1., .5]
v
    # v1: [.83, .55]^T
    # v2: [-.71, .71]^T

# PC computation
#   i) subtract x_bar from each vector x.
#   ii) find eigenvalue and (unit) eigenvector using
#       a) covariance matrix;
#       b) singular value decomposition.
#           X = USV^T, where
#           .) the columns of U are left singular vectors (eigenvectors);
#           ..) the columns of V are right singular vectors;
#           ...) the diagonal entries of S are singular values (sqrt(eigenvalues)).
#   iii) project data onto PCs
#       e.g. let v1 be the 1st PC, then X*v1 will give observations'
#       value on the 1st PC.
#       in general, we build a **transformation matrix**, the columns
#       of which are eigenvectors.

# Visualization: iris (4D to 2D)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
    # import utilities
data = load_iris() # class marker: 0, 1, 2
y = data.target
X = data.data
    # load data
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X)
    # build model
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(reduced_X)):
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()


## PCA: Application (facial recognition)

from os import walk, path
import numpy as np
import mahotas as mh
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
    # import utilities
X = []
y = []
for dir_path, dir_name, file_names in walk('orl_faces'):
    for fn in file_names:
        if fn[-3:] == 'pgm':
            image_filename = path.join(dir_path, fn)
            X.append(scale(mh.imread(image_filename,
                    as_grey=True).reshape(10304).astype('float32')))
            y.append(dir_path)
X = np.array(X)
X_train,X_test,y_train,y_test = train_test_split(X,y)
    # loading data
pca = PCA(n_components=150)
    # 400 units of 10304D observations reduced to 150D.
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)
print 'The original dimensions of the training data were',\
    X_train.shape
print 'The reduced dimensions of the training data are',\
    X_train_reduced.shape
classifier = LogisticRegression()
accuracies = cross_val_score(classifier, X_train_reduced, y_train)
    # [ 0.78761062,  0.82524272,  0.89285714]
print 'Cross validation accuracy:', np.mean(accuracies), accuracies
    # 0.835236826924 [ 0.78761062  0.82524272  0.89285714]
classifier.fit(X_train_reduced, y_train)
predictions = classifier.predict(X_test_reduced)
print classification_report(y_test, predictions)



















































