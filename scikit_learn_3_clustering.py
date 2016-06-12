### CLUSTERING


## K-Means

# cost function
#   J = \sum_k\sum_{i\in C_k} ||x_i - \mu_k||^2
#   i.e. sum of squares of Euclidean distance of
#   observations to centroids.

# KM algorithm
#   i) choose k, randomly select k initializer centroids;
#   ii) assign all observations to centroids by Euclidean distance;
#   iii) adjust centroids by observations assigned to it;
#   iv) unless tired, go to ii), until centroids stop moving.

# KM demo on toy data (randomly generated, TODO: DATA GENERATION FIXING)
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
    # import utilities
cluster1 = np.random.uniform(.5,1.5,(2,10))
cluster2 = np.random.uniform(3.5,4.5,(2,10))
    # np.random.uniform(a,b,(<# of sets>,<# of obs in sets>))
X = np.hstack((cluster1,cluster2)).T
X = np.vstack((cluster1,cluster2)).T
    # np.hstack: horizontal stacking
    # x = np.array([1,2,3]); y = np.array([3,4,5])
    # np.hstack((x,y))
    #   array([1, 2, 3, 3, 4, 5])
    # np.vstack((x,y))
    #   array([[1, 2, 3],
    #          [3, 4, 5]])
K = range(1,10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X,kmeans.cluster_centers_,
                                            'euclidean'),axis=1)) / X.shape[0])
    # model building (for K=1,...,9)
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
plt.show()


## Evaluation

# silouette coefficient
#   s = ba/max(a,b)
#       a: mean dist between instances in a cluster.
#       b: mean dist between the instances in the cluster
#           and the instances in the next closest cluster.
#   a measure of the compactness and separation of
#   the clusters. it is calculated for each instance,
#   for a data set, it is calculated as the mean of
#   the individual samples' scores.
#   value range: [0,1]; the higher the better.
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
plt.subplot(3,2,1)
    # subplot(nrows,ncols,fignum)
    #   subplot(211) = subplot(2,1,1)
    #   fignum: position of current picture.
x1 = np.array([1,2,3,1,5,6,5,5,6,7,8,9,7,9])
x2 = np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])
    # len: 14
X = np.array(zip(x1,x2)).reshape(len(x1),2)
    # zip(..,..): 14 by 2 matrix
plt.xlim([0,10])
plt.ylim([0,10])
plt.title('Instances')
plt.scatter(x1,x2)
colors = ['b','g','r','c','m','y','k','b']
markers = ['o','s','D','v','^','p','*','+']
tests = [2,3,4,5,8]
subplot_counter = 1
for t in tests:
    subplot_counter += 1
    plt.subplot(3,2,subplot_counter)
    kmeans_model = KMeans(n_clusters=t).fit(X)
    for i, l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i],x2[i],color=colors[l],marker=markers[l],
                 ls='None')
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.title('K = %s, silouette coefs = %.03f' %
              (t, metrics.silhouette_score(X,kmeans_model.labels_,
                                          metric='euclidean')))
plt.show()


## Image Compression

# image quantization
#   a lossy compression method that replaces a range
#   similar colors (within-cluster members) in an image
#   with a single color. it reduces the size of the image file.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import mahotas as mh
    # import utilities
original_img = np.array(mh.imread('image_quantization.png'),
                        dtype=np.float64) / 255
original_dimensions = tuple(original_img.shape)
width, height, depth = tuple(original_img.shape)
image_flattened = np.reshape(original_img,(width*height, depth))
    # flatten image
image_array_sample = shuffle(image_flattened,
                             random_state=0)[:1000]
estimator = KMeans(n_clusters=64, random_state=0)
estimator.fit(image_array_sample)
    # create 64 clusters from a sample of 1000 randomly
    # selected colors.
cluster_assignments = estimator.predict(image_flattened)
compressed_palette = estimator.cluster_centers_
compressed_img = np.zeros((width,height,compressed_palette.shape[1]))
label_idx = 0
for i in range(width):
    for j in range(height):
        compressed_img[i][j] = compressed_palette[cluster_assignments[label_idx]]
        label_idx += 1
    # fill the cluster centroids to compressed image by width and height
plt.subplot(122)
plt.title('Original')
plt.imshow(original_img)
plt.axis('off')
plt.subplot(121)
plt.title('Compressed')
plt.imshow(compressed_img)
plt.axis('off')
plt.show()


## Semi-Supervised Learning: Cluster + Classification

# procedure:
#   i) learn features by clustering unlabeled data.
#       (Phase I and II)
#   ii) use learned features to build a supervised classifier.
#       (Phase III)

# Phase I: clustering
#   representation of images
#       i) each image is broken down into some descriptors which
#           which be clustered.
#       ii) the image is then represented with a vector with
#           one element for each cluster (each element will
#           encode the number of descriptors extracted from
#           the image that were assigned to the cluster).
#   current test
#       train .6; test .4
#   at Phase I, we do i).
import numpy as np
import mahotas as mh
from mahotas.features import surf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.cluster import MiniBatchKMeans
    # MiniBatchKMeans
    #   computes the distances to the centroids for
    #   only a sample of the instances in each iteration.
    #   the benefit is faster convergence.
import glob
    # import utilities
all_instance_filenames = []
all_instance_targets = []
for f in glob.glob('cat_dog_images/*.jpg'):
    target = 1 if 'cat' in f else 0
    all_instance_filenames.append(f)
    all_instance_targets.append(target)
    # sort images by indexing cat 1 dog 0
surf_features = []
counter = 0
for f in all_instance_filenames:
    print 'Reading image:', f
    image = mh.imread(f, as_grey=True)
    surf_features.append(surf.surf(image)[:,5:])
    # only save some of the descriptors
train_len = int(len(all_instance_filenames)*.6)
X_train_surf_features = np.concatenate(surf_features[:train_len])
X_test_surf_features = np.concatenate(surf_features[train_len:])
y_train = all_instance_targets[:train_len]
y_test = all_instance_targets[train_len:]
    # training-testing split
n_clusters = 300
print 'Clustering', len(X_train_surf_features), 'features'
estimator = MiniBatchKMeans(n_clusters=n_clusters)
estimator.fit_transform(X_train_surf_features)
    # model building

# Phase II: feature learning
#   at Phase II we do ii).
#   we end up with a list of images, each is represented
#   as a bincounted list of cluster numberings -- for instance,
#       feature 1   feature 2   ...   feature n
#   [   25          340         ...   210       ]
X_train = []
for instance in surf_features[:train_len]:
    # each instance is a surf_feature representation of an image.
    clusters = estimator.predict(instance)
    # a 'clusters' object is a after-clustering feature
    #   vector of the image.
    features = np.bincount(clusters)
    # bincount is a list of length n_clusters (300 in this case).
    #   in each bin is the number of ith feature the image have.
    if len(features) < n_clusters:
        features = np.append(features,
                             np.zeros((1,n_clusters-len(features))))
    X_train.append(features)
    # X_train ends up as a list of lists bincounted features.
X_test = []
for instance in surf_features[train_len:]:
    clusters = estimator.predict(instance)
    features = np.bincount(clusters)
    if len(features) < n_clusters:
        features = np.append(features,
                             np.zeros((1,n_clusters-len(features))))
    X_test.append(features)
    # same operation on X_test

# Phase III: classification
clf = LogisticRegression(C=.001, penalty='l2')
clf.fit_transform(X_train, y_train)
predictions = clf.predict(X_test)
print classification_report(y_test, predictions)
print 'Precision: ', precision_score(y_test, predictions)
print 'Recall: ', recall_score(y_test, predictions)
print 'Accuracy: ', accuracy_score(y_test, predictions)
    # performance-wise, this outperforms classifiers that
    # use only the pixel intensities as features, with much
    # lower dimensional features too (300 here, 10,000 for
    # even small images with 100 by 100 pixels.
















































