### SVM

# other nonlinear models
#   polynomial regression, for instance, involves using combination
#   of features to create new features.
#   the high-dimensionality representations requires exponentially more
#   training data to avoid overfitting.

## SVM I (Basics)

# similar mechanism as Perceptron, which has
#   the following components:
#   y = act_func(argument)
#   i) argument: a linear regression model
#       \sum_{i}^{p} w_i*x_i + b
#       where x_i are variables, b is intercept/bias term.
#   ii) act_func: activation function
#       e.g. 1, Heaviside Step Function
#           g(arg) = {1, if arg>0; 0, otherwise}
#       e.g. 2, Logistic Function
#           g(arg) = 1/(1+e^{arg})

# SVM
#   y = func(argument)
#   i) argument: a linear regression model
#       f(x) = <w,x> + b
#       <,> denotes inner product.
#   ii) func: activation function
#       func(x) = sign(f(x))

# form
#   i) primal: f(x) = <w,x> + b
#   ii) dual: f(x) = \sum \alpha_i*y_i*<x_i,x> + b
#       inner product of training instances & test instances'
#       feature vector.

# high-D mapping (plain)
#   let \phi be a function such that
#   i) x -> \phi(x)
#   ii) \phi: R^d -> R^D
#   this incorporated in the dual form, produces
#   f(x) = \sum \alpha_i*y_i*<\phi(x_i),\phi(x)> + b
#   problem: high computational cost in computing dot product

# high-D mapping (kernel)
#   kernel method doesn't explicity compute dot product,
#   but can produce the same values.
#   K(x,z) = <\phi(x),\phi(z)>
#   demo:
#       x = (x_1,x_2); z = (z_1,z_2); \phi(x) = x^2
#       i) plain high-D mapping
#           <\phi(x),\phi(z)>
#           = <(x_1^2,x_2^2,\sqrt(2)x_1x_2),(z_1^2,z_2^2,\sqrt(2)z_1z_2)> = ret1
#       ii) kernel trick
#           K(x,z) = <x,z>^2 = (x_1z_1+x_2z_2)^2 = ret2
#       ret1 = ret2!
#   kernel does the same thing with fewer arithmetic operations.

# types of kernel
#   i) polynomial: (\gamma*<x_i,x_j>+r)^d
#   ii) Gaussian: exp(-\gamma*|x_i-x_j|^2)
#   iii) sigmoid: tanh(<x_i,x_j>+r)


## SVM II (Max Margin Classifier)

# max margin
#   i) function margin (FM):
#       funct = min y_i*f(x_i)
#           where f(x) = <w,x> + b
#       if correctly classified, FM is large, otherwise small.
#   ii) support vector
#       if FM = 1, the instance in question is a support vector (SV).
#       SV's alone are sufficient to define decision boundary.
#   iii) geometric margin
#       the maximum width of the band that separates the SV's.
#       **optimization for this is beyond the scope**


## SVM III (handwritten chars recognition)

# data demo
#   Mixed National Institute of Standards and Technology
#       70,000 images, 28 by 28 in size.
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import matplotlib.cm as cm
    # import utilities
digits = fetch_mldata('MNIST original',data_home='data/mnist').data
counter = 1
for i in range(1,4):
    for j in range(1,6):
        plt.subplot(3,5,counter)
        plt.imshow(digits[(i-1)*8000+j].reshape((28,28)),
                   cmap=cm.Greys_r)
        plt.axis('off')
        counter += 1
plt.show()

# svm
from sklearn.datasets import fetch_mldata
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
    # import utilities
data = fetch_mldata('MNIST original',data_home='data/mnist')
X,y = data.data,data.target
X = X/255.*2-1
X_train,X_test,y_train,y_test = train_test_split(X,y)
    # load data
pipeline = Pipeline([
    ('clf', SVC(kernel='rbf',gamma=.01,C=100))
    # kernel: radial basis function
])
print X_train.shape
parameters = {
    'clf__gamma': (.01,.03,.1,.3,1.),
    'clf__C': (.1,.3,1.,3.,10.,30.)
}
grid_search = GridSearchCV(pipeline,parameters,
                           n_jobs=2,verbose=1,
                           scoring='accuracy')
grid_search.fit(X_train[:10000],y_train[:10000])
print 'Best score: %.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t%s: %r' % (param_name,best_parameters[param_name])
predictions = grid_search.predict(X_test)
print classification_report(y_test,predictions)


## SVM IV (regular image classification)

# chars74k data
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import Image
    # import utilities
def resize_and_crop(image,size):
    img_ratio = image.size[0]/float(image.size[1])
    ratio = size[0]/float(size[1])
    if ratio > img_ratio:
        image = image.resize((size[0],size[0]*image.size[1]/
                             image.size[0]),Image.ANTIALIAS)
        image = image.crop((0,0,30,30))
    elif ratio < img_ratio:
        image = image.resize((size[1]*image.size[0]/
                              image.size[1],size[1]),Image.ANTIALIAS)
        image = image.crop((0,0,30,30))
    else:
        image = image.resize((size[0],size[1]),Image.ANTIALIAS)
    return image
    # resize function
X = []
y = []
for path, subdirs, files in os.walk('English/Img/GoodImg/Bmp'):
    for filename in files:
        f = os.path.join(path,filename)
        img = Image.open(f).convert('L')
            # convert to greyscale
        img_resized = resize_and_crop(img,(30,30))
        img_resized = np.asarray(img_resized.getdata(),
                                 dtype=np.float64).\
            reshape((img_resized.size[1]*img_resized.size[0],1))
        target = filename[3:filename.index('-')]
        X.append(img_resized)
        y.append(target)
X = np.array(X)
X = X.reshape(X.shape[:2])
    # train svm with a polynomial kernel
classifier = SVC(verbose=0,kernel='poly',degree=3)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)
classifier.fit(X_train,y_train)
predictions = classifier.predict(X_test)
print classification_report(y_test,predictions)



















