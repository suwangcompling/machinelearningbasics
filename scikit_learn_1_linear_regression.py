### LINEAR REGRESSION

## Linear Regression I (Univariate)

# import utilities
import sklearn
    # main installation:
    #   i) pip install -U scikit-learn
    #   ii) pip install -U numpy
    #   iii) pip install -U scipy
    #   restart python console
import pandas as pd
import matplotlib.pyplot as plt
    # other installation:
    #   i) pip install pandas
    #   ii) pip install matplotlib

# put data into matrix with pandas.DataFrame
d = {'diameter': [6., 8., 10., 14., 18.], \
     'price': [7., 9., 13., 17.5, 18.]}
data = pd.DataFrame(d)
X = data.diameter
y = data.price
plt.figure()
plt.title('Pizza')
plt.xlabel('diameter')
plt.ylabel('price')
plt.plot(X,y,'k.')
plt.axis([0,25,0,25])
plt.grid(True)
plt.show()

# linear regression fitting
from sklearn.linear_model import LinearRegression
X = [[data.diameter[i]] for i in range(len(data.diameter))]
y = [[data.price[i]] for i in range(len(data.price))]
    # create data in the form e.g. [[6.0], [8.0], [10.0], [14.0], [18.0]]
model = LinearRegression()
model.fit(X,y)
    # beta_0: model.intercept_ = 1.96551724
    # beta_1: model.coef_ = 0.9762931
    # y_hat = 1.97 + 0.98X
model.predict(12)[0]
    # predict: how much does a 12" pizza cost


# visualization
plt.plot(X,y,'k.')
plt.axis([0,25,0,25])
plt.plot(X,model.predict(X))

# performance evaluations
import numpy as np
rss = np.sum((model.predict(X)-y)**2)
    # residual sum of squares
mse = (1.0/len(X)) * rss
    # mean sum of squares
var = np.var(X,ddof=1)
    # variance
    # ddof: degree of freedom, N - ddof
cov = np.cov([X[i][0] for i in range(len(X))],\
             [y[i][0] for i in range(len(y))])
    # covariance matrix (2 by 2)
    # the off-diagonal values are X-y covariance
    # on-diagonal: XX, var of X; yy, var of y
from __future__ import division
coef = cov[0][1] / var
    # note that beta_1 = cov / var
    # beta_0 can be computed as follows:
    #   beta_0 = y_bar - beta_1*x_bar
r_sq_train = model.score(X,y)
    # r-squared
    #   training r^2: 0.91000159642401024
r_sq_test = model.score([[8], [9], [11], [16], [12]], [[11], [8.5], [15], [18], [11]])
    #   test r^2: 0.6620052929422553

## Linear Regression II (Multivariate)

# put data in matrix
d1 = {'diameter': [6., 8., 10., 14., 18.], \
     'num_toppings': [2., 1., 0., 2., 0.], \
     'price': [7., 9., 13., 17.5, 18.]}
d2 = {'diameter': [8., 9., 11., 16., 12.], \
      'num_toppings': [2., 0., 2., 2., 0.], \
      'price': [11., 8.5, 15., 18., 11.]}
train = pd.DataFrame(d1)
test = pd.DataFrame(d2)

# multivariate lin. reg. coefficient
#   Beta = (X^t X)^{-1} X^t Y
#   manual computation:
#       from numpy.linalg import inv
#       from numpy import dot, transpose
X = np.matrix(train.iloc[:,[0,1]])
y = np.matrix(train.iloc[:,[2]])
model = LinearRegression()
model.fit(X,y)
predictions = model.predict(X)
    # prediction on training, for demo
r_sq_train = model.score(X,y)

## Linear Regression III (Polynomial)

# import utilities
from sklearn.preprocessing import PolynomialFeatures

# load data
X = [[6],[8],[10],[14],[18]]
y = [[7],[9],[13],[17.5],[18]]

# model building
#   y = beta_0 + beta_1*x + beta_2*x^2
quadratic_featurizer = PolynomialFeatures(degree=2)
X_quad = quadratic_featurizer.fit_transform(X)
model = LinearRegression()
model.fit(X_quad,y)
    # plotting: location 827/4137

## Linear Regression IV (Real Data Demo: Wine)

# import utilities
import pandas as pd

# load data
df = pd.read_csv('wine_data.csv')
df.columns = ['quality','alcohol','malic_acid','ash','alcalinity_of_ash',\
              'magnesium','total_phenols','flavanoids','nonflavanoid_phenols',\
              'proanthocyanins','color_intensity','hue',\
              'od280/od315_of_diluted_wines','proline']

# visualization (alcohol against quality)
import matplotlib.pylab as plt
plt.scatter(df['alcohol'],df['quality'])
plt.scatter(df['alcohol'],df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol against Quanlity')
plt.show()

# model building
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
X = df[list(df.columns)[1:]]
    # first column is the response: quality
y = df['quality']
X_train,X_test,y_train,y_test = train_test_split(X,y)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# evaluation
r_sq_test = model.score(X_test,y_test)
    # 0.796!
from sklearn.cross_validation import cross_val_score
model = LinearRegression()
scores = cross_val_score(model,X,y,cv=5)
print scores.mean() # cv mean
print scores # cv for five sets
    # cross-validation


### GRADIENT DESCENT

# Methods
#
# cost function: RSS
#   \sum_{1}^{n}(y_i - f(x_i))^2
#       local minimum
#
# types
#   batch GD: use all training instances to update parameters
#       at each iteration. it is deterministic -- producing
#       the same parameter values with a given training set.
#   stochastic GD: use 1 (randomly chosen) training instance to
#       update parameters at each iteration. it is nondeterministic,
#       produces different estimates at different runs.

# SGD Demo

# import utilities
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

# load & scaling data
data = load_boston()
X_train,X_test,y_train,y_test = \
    train_test_split(data.data,data.target)
X_scaler, y_scaler = StandardScaler(), StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)
X_test = X_scaler.fit_transform(X_test)
y_test = y_scaler.fit_transform(y_test)

# model building
model = SGDRegressor(loss='squared_loss')
scores = cross_val_score(model,X_train,y_train,cv=5)
print 'CV r-squared: ', scores
print 'Avg CV r-squared: ', np.mean(scores)
model.fit_transform(X_train,y_train)
print  'Test set r-squared: ', model.score(X_test,y_test)
    # CV r-squared:  [ 0.75474391  0.61151436  0.69517402  0.71785126  0.41502689]
    # Avg CV r-squared:  0.638862088204
    # Test set r-squared:  0.822425952544


### FEATURE EXTRACTION & PREPROCESSING

## Feature Extraction

# Example 1: vectorize cities
#   DictVectorizer

from sklearn.feature_extraction import DictVectorizer
onehot_encoder = DictVectorizer()
instances = [
    {'city': 'New York'},
    {'city': 'San Francisco'},
    {'city': 'Chapel Hill'}
]
onehot_encoder.fit_transform(instances).toarray()
    # vectorize alphabetically
    #    array([[ 0.,  1.,  0.],
    #           [0., 0., 1.],
    #           [1., 0., 0.]])

# Example 2: bag-of-words representation
#   CountVectorizer

# toy corpus
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]

# feature extraction
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
print vectorizer.fit_transform(corpus).todense()
    # [[0 1 1 0 1 0 1 0 0 1]
    #  [0 1 1 1 0 1 0 0 1 0]
    #  [1 0 0 0 0 0 0 1 0 0]]
    # 10 words in the vocabulary
    # a word corresponds to 1 if it presents in a doc
print vectorizer.vocabulary_
    # {u'duke': 2, u'basketball': 1, u'lost': 5,
    # u'played': 6, u'in': 4, u'game': 3, u'sandwich': 7,
    # u'unc': 9, u'ate': 0, u'the': 8}

# distance between docs
#   Euclidean: ||x|| = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
from sklearn.metrics.pairwise import euclidean_distances
counts = vectorizer.fit_transform(corpus).todense()
one_two_dist = euclidean_distances(counts[0],counts[1])
    # array([[ 2.44948974]])

# issues:
# 1. sparsity: if docs are long and numerous, many will
#       have a large number of 0 cells.
#       sol: this will be tackled by representing
#           only nonzero elements somehow.
# 2. Hughes effect: as the feature space dimensionality
#       increase, more training data is required to
#       ensure model doesn't overfit (n >> p).
#       sol: dimensionality reduction.

# Example 3: eliminate stop-words

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]
vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit_transform(corpus).todense()
print vectorizer.vocabulary_
    # {u'duke': 2, u'basketball': 1, u'lost': 4,
    # u'played': 5, u'game': 3, u'sandwich': 6,
    # u'unc': 7, u'ate': 0}
    # 'the', for instance, is omitted.

# Example 4: Stemming & Lemmatization (SL)
#   L: obtain morphological roots of words
#   S: remove affixes only

# before SL
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'He ate the sandwiches',
    'Every sandwich was eaten by him'
]
vectorizer = CountVectorizer(binary=True,stop_words='english')
vectorizer.fit_transform(corpus).todense()
print vectorizer.vocabulary_
    # {u'sandwich': 2, u'ate': 0, u'sandwiches': 3, u'eaten': 1}

# after SL
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print lemmatizer.lemmatize('gathering','v') # gather
print lemmatizer.lemmatize('gathering','n') # gathering
    # demo for single word lemmatization
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
print stemmer.stem('gathering') # gather
    # demo for single word stemming

# LS toy corpus
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
wordnet_tags = ['n','v']
corpus = [
    'I am gathering ingredients for the sandwich.',
    'There were many wizards at the gather.'
]
stemmer = PorterStemmer()
print 'Stemmed: ', [[stemmer.stem(token) for token in
                     word_tokenize(document)] for document in corpus]
    # stemming
def lemmatize(token,tag):
    if tag[0].lower() in ['n','v']:
        return lemmatizer.lemmatize(token,tag[0].lower())
    return token
lemmatizer = WordNetLemmatizer()
tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]
print 'Lemmatized: ', [[lemmatize(token,tag) for (token,tag) in document]
                       for document in tagged_corpus]


# TF-IDF

# documents defined by freq_dict of words
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['The dog ate a sandwich, the wizard transfigured a sandwich, '
          'and I ate a sandwich']
vectorizer = CountVectorizer(stop_words='english')
doc_vector = vectorizer.fit_transform(corpus).todense()
    # matrix([[2, 1, 3, 1, 1]])
doc_freq_dict = vectorizer.vocabulary_
    # {u'ate': 0, u'dog': 1, u'sandwich': 2, u'transfigured': 3, u'wizard': 4}

# normalizing term-frequency (tf)
#   why? with the same tf, larger documents might be very
#   different from small documents!
#   i) normalize by size of doc:
#       tf(t,d) = [f(t,d)+1] / ||x||
#   ii) normalize by log-transformation (sublinear_tf=True):
#       tf(t,d) = log(f(t,d)+1)
#   iii) most-freq-word normalization:
#       .5 + [.5*f(t,d)] / max_f(w,d):w\in d

# idf (inverse document frequency)
#   meaures: how rare or common a word is in a corpus
#   idf(t,D) = log(N/(1+|d\in D:t\in d|))
#   if t is super rare, then denom in log(*) is small, * is large
#   idf(t,D) is thus large!

# tf-idf measure
#   tf-idf = tf * idf (use_idf=True)
#   i.e. high normalized freq & rare = high tf-idf!

# tf-idf weighted feature vectors
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ['The dog ate a sandwich and I ate a sandwich',
          'The wizard transfigured a sandwich']
vectorizer = TfidfVectorizer(stop_words='english')
doc_tfidf_freq_dict = vectorizer.fit_transform(corpus).todense()


# FEATURIZING IMAGES

# stupid approach
from sklearn import datasets
digits = datasets.load_digits()
digit_index =  digits.target[0]
digit_matrix = digits.images[0]
    # image defined by greyscale, 8 by 8
digit_feature = digit_matrix.reshape(-1,64)
    # know that this way of featurizing image
    # is sensitive to minute changes in image
    # e.g. shifting, rotation, zooming, etc.

# revision: point of interest
#   i) edges: where pixel intensity rapidly changes
#   ii) corners: where two edges intersect
import numpy as np
from skimage.feature import corner_harris, corner_peaks
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.exposure import equalize_hist
    # import utilities
def show_corners(corners,image):
    fig = plt.figure()
    plt.gray()
    plt.imshow(image)
    y_corner, x_corner = zip(*corners)
    plt.plot(x_corner, y_corner, 'or')
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    fig.set_size_inches(np.array(fig.get_size_inches())*1.5)
    plt.show()
    # corner detector
mandrill = io.imread('image_detect.png')
mandrill = equalize_hist(rgb2gray(mandrill))
corners = corner_peaks(corner_harris(mandrill),min_distance=2)
show_corners(corners,mandrill)

# advanced image featurization methods
#   i) Scale-Invariant Feature Transform (SIFT)
#       describes edges and corners
#   ii) Speeded-Up Robust Features (SURF)
#       descriptions that are invariant of scale, orientation, and illumination
import mahotas as mh
from mahotas.features import surf
    # import utilities
image = mh.imread('image_detect_2.jpg')
image = image[:,:,0] # convert to grayscale
surf_descriptor_1 = surf.surf(image)[0]
num_descriptors = len(surf.surf(image))
    # a SURF demo

# DATA STANDARDIZATION

# N(0,1)
from sklearn import preprocessing
import numpy as np
X = np.array([
    [0.,0.,5.,13.,9.,1.],
    [0.,0.,13.,15.,10.,15.],
    [0.,3.,15.,2.,0.,11.]
])
X_norm = preprocessing.scale(X)
    # [[ 0.         -0.70710678 -1.38873015  0.52489066  0.59299945 -1.35873244]
    #  [ 0.         -0.70710678  0.46291005  0.87481777  0.81537425  1.01904933]
    #  [ 0.          1.41421356  0.9258201  -1.39970842 -1.4083737   0.33968311]]






