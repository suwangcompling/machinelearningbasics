### CLASSIFICATION


## Spam Filtering

# data preparation
import pandas as pd
df = pd.read_csv('SMSSpamCollection',delimiter='\t',header=None)
num_spam = df[df[0]=='spam'][0].count() # 747
num_ham = df[df[0]=='ham'][0].count() # 4825

# model building
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import preprocessing
    # import utilities
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1],df[0])
    # train_test_split(predictors, response)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
    # now texts are represented as tf-idf vectors
    # fit_transform vs. transform
    '''
    Fit and then transform, make it possible to fit on training data
    and transform on test data. So the transformation on training
    and testing data set are identical, and the transformed data can be
    compared and computed together.

    If you use fit_transform in two data set, then there would be two
    different coordinate system for the transformed data. So the transformed
    result cannot be computed or computed together.
    '''
classifier = LogisticRegression()
label_fixer = preprocessing.LabelBinarizer()

label_fixer.fit(y_train)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
# for i, prediction in enumerate(predictions[:5]):
#     print 'Prediction: %s. Message: %s' % (prediction, X_test_raw[i])
    # print some sample ham/spam classification results
    # CODE INVALID

# evaluation: toy examples
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
y_test = [0,0,0,0,0,1,1,1,1,1]
y_pred = [0,1,0,0,0,0,0,1,1,1]
confusion_matrix = confusion_matrix(y_test,y_pred)
    #    0   1
    # 0  tn  fp
    # 0  fn  tp
    # array([[4, 1],
    #        [2, 3]])
from sklearn.metrics import accuracy_score
y_pred, y_true = [0,1,1,0], [1,1,1,1]
accuracy_score(y_true, y_pred)
    # accuracy: 0.5
    # problem of accuracy as an evaluation metric:
    #   disregard fp & fn errors, while the distinction
    #   may be meaningful for some tasks

# example: spam data
scores = cross_val_score(classifier, X_train, y_train, cv=5)
mean_accuracy, cv_accuracies =  np.mean(scores), scores
    # accuracy:
    # 0.955013010114 [ 0.95459976  0.96052632  0.95095694  0.96167665  0.94730539]
lb = preprocessing.LabelBinarizer()
y_train = np.array([number[0] for number in lb.fit_transform(y_train)])
    # response binarization: necessary for prec, rec, roc, auc!
precisions = cross_val_score(classifier, X_train, y_train,\
                             cv=5, scoring='precision')
mean_precision = np.mean(precisions)
    # precision: tp(tp+fp)
    # 0.98696946068875879 [ 0.96052632,  0.98666667,  1.        ,  0.98765432,  1.        ]
    # ~= .02 fp (i.e. spam accepted as ham)
recalls = cross_val_score(classifier, X_train, y_train,\
                          cv=5, scoring='recall')
mean_recall = np.mean(recalls)
    # recall: tp/(tp+fn)
    # 0.685714285714 [ 0.65178571  0.66071429  0.67857143  0.71428571  0.72321429]
    # ~= .32 fn (i.e. ham rejected as spam)
f1s = cross_val_score(classifier, X_train, y_train, \
                      cv=5, scoring='f1')
mean_f1s = np.mean(f1s)
    # f1 = 2* (prec*rec / (prec+rec))
    # 0.808988803126 [ 0.77659574  0.79144385  0.80851064  0.82901554  0.83937824]

# Curve Evaluation

# Receiver Operating Characteristic (ROC)
#   fall-out (x) against recall (y)
#   fall-out = fp/(tn+fp), the rate of misidentified negs out of all negs
# Area Under Curve (AUC)
#   ROC reduced to 1 value
from sklearn.metrics import roc_curve, auc
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1],df[0])
    # reload data (toy examples changed some of them)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
y_test = np.array([number[0] for number in lb.fit_transform(y_test)])
    # binarization
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict_proba(X_test)
fp_rate, recall, thresholds = roc_curve(y_test, predictions[:,1])
roc_auc = auc(fp_rate, recall)
    # 0.99248113602551424
plt.plot(fp_rate, recall, 'b', label='AUC=%.2f' % roc_auc) # actual line
plt.plot([0,1],[0,1],'r--') # random-guess dash
plt.title('ROC')
plt.legend(loc='lower right')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')


## Hyperparameters
#   parameters of the model that are not learned

# Grid Search (TODO: CODE NEEDS FIXING)
#   takes a set of possible values for each hyperparameter
#   that should be tuned, and evaluate a model trained on each
#   element o the Cartesian product of the sets.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
    # import utilities
pipeline = Pipeline([
    ('vect',
     TfidfVectorizer(stop_words='english')),
    ('clf',LogisticRegression())
])
parameters = {
    'vect__max_df': (.25,.5,.75),
    'vect__stop_words': ('english',None),
    'vect__max_features': (2500,5000,10000,None),
    'vect__ngram_range': ((1,1),(1,2)),
    'vect__use_idf': (True,False),
    'vect__norm': ('l1','l2'),
    'clf_penalty': ('l1','l2'),
    'clf__C': (.01,.1,1,10)
}
if __name__ == '__main__':
    grid_search = GridSearchCV(pipeline, parameters,\
                               n_jobs=-1,verbose=1,\
                               scoring='accuracy',cv=3)
        # n_jobs: max concurrent jobs
    import pandas as pd
    df = pd.read_csv('SMSSpamCollection', delimiter='\t', header=None)
    X, y, = df[1], df[0]
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X)
    X_test = vectorizer.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    grid_search.fit(X_train,y_train)
    print 'Best score: %.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    predictions = grid_search.predict(X_test)
    print 'Accuracy:', accuracy_score(y_test, predictions)
    print 'Precision:', precision_score(y_test, predictions)
    print 'Recall:' , recall_score(y_test, predictions)


## Multiclass Classification (TODO: CODE NEEDS FIXING)

# data
#   Rotten Tomatoes Movie Reviews
#   https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data?

# data exploration
import pandas as pd
df = pd.read_csv('train.tsv', header=0, delimiter='\t')
print df.count() # 156060 instances
print df.head()
    # shape: 156060 by 4
    # parameters: phraseId, SentenceId, Phrase, Sentiment
    # response gradation:
    #   0 (neg), 1 (somewhat neg), 2 (neutral),
    #   3 (somewhat positive), 4 (positive)
print df['Sentiment'].value_counts()
    # print out count of reviews in each response type

# model building
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

def run():
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(stop_words='eng')),
        ('clf', LogisticRegression())
    ])
    parameters = {
        'vect__max_df': (.25, .5),
        'vect__ngram_range': ((1,1),(1,2)),
        'vect_use_idf': (True,False),
        'clf__C': (.1, 1, 10),
    }
df = pd.read_csv('train.tsv', header=0, delimiter='\t')
X, y = df['Phrase'], df['Sentiment'].as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=.5)
grid_search = GridSearchCV(pipeline,parameters,
                            n_jobs=3,verbose=1,
                            scoring='accuracy')
grid_search.fit(X_train, y_train)
print 'Best score: %.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t%s: %r' % (param_name, best_parameters[param_name])


## Multilabel Classification

# problem transformation
#   each observation can be classified as A,B,C,...
#   and can carry multiple labels.
#   build multiple classifiers to run through all observations.
#   performance metrics:
#       i) Hamming loss: average fraction of incorrect labels.
#       ii) Jaccard similarity:
#           J(pred,true) = (pred\intersect true) / (pred\union true)
import numpy as np
from sklearn.metrics import hamming_loss, jaccard_similarity_score
pred = np.array([[0.,1.],[1.,1.]])
true = np.array([[0.,1.],[1.,1.]])
hamming_loss(pred,true) # 0
jaccard_similarity_score(pred,true) # 1
    # in general, build binary logistic models as before,
    # and evaluate each model with hamming & jaccard
    # and take average over all models.


## Decision Trees

# entropy
#   H(X) = -\sum_{i}(p(x_i)log_2(p(x_i)))
#   e.g. coin: -(.5*log_2(.5) + .5*log_2(.5))

# info gain (pp104, 2135/4137)
#   IG(T,a) = H(T) - \sum_{a} [|{x\in T|x_a = v}|/|T|] *
#             H({x\in T|x_a = v})
#   the difference between entropy of parent
#   and weighted avg entropy of children.

# use of entropy and info gain in DT
#   i) at each split, compute info gain for each variable
#       and pick the one with the highest IG.
#   ii) iterate until variables have been used up.

# alternative to IG: Gini impurity
#   Gini(t) = 1 - \sum_{i=1}^{p} p(i|t)^2
#   i: class i; t: the set of observations; p: # of classes
#   Gini impurity reaches peak when all of the elements
#   of the set are the same class.
#   Gini_max = 1 - 1/p

# picture classification (ad or not)
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
    # import utilities
df = pd.read_csv('ad.data',header=None)
explanatory_variable_columns = set(df.columns.values)
response_variable_column = df[len(df.columns.values)-1]
explanatory_variable_columns.remove(len(df.columns.values)-1)
y = [1 if e=='ad.' else 0 for e in response_variable_column]
X = df[list(explanatory_variable_columns)]
X.replace(to_replace=' *\?', value=-1, regex=True, inplace=True)
X_train,X_test,y_train,y_test = train_test_split(X,y)
pipeline = Pipeline([
    ('clf', DecisionTreeClassifier(criterion='entropy'))
])
parameters = {
    'clf__max_depth': (150,155,160),
    'clf__min_samples_split': (1,2,3),
    'clf__min_samples_leaf': (1,2,3)
}
grid_search = GridSearchCV(pipeline,parameters,n_jobs=-1,
                            verbose=1,scoring='f1')
grid_search.fit(X_train,y_train)
    # find the best configuration of hyperparameters.
print 'Best score: %.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t%s: %r' % (param_name,best_parameters[param_name])
predictions = grid_search.predict(X_test)
print classification_report(y_test,predictions)


## Tree Emsemble

# Random Forest
#   a collection of decision trees that have been trained
#   on randomly selected subsets of the training instances
#   and explanatory variables.
#   make predictions by returning the mode or mean of the
#   predictions of their constituent trees. (sklearn: mean)
#   beats plain DT by avoiding overfitting.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
    # import utilities
df = pd.read_csv('ad.data',header=None)
explanatory_variable_columns = set(df.columns.values)
response_variable_column = df[len(df.columns.values)-1]
explanatory_variable_columns.remove(len(df.columns.values)-1)
y = [1 if e=='ad.' else 0 for e in response_variable_column]
X = df[list(explanatory_variable_columns)]
X.replace(to_replace=' *\?', value=-1, regex=True, inplace=True)
X_train,X_test,y_train,y_test = train_test_split(X,y)
pipeline = Pipeline([
    ('clf', RandomForestClassifier(criterion='entropy'))
])
parameters = {
    'clf__n_estimators': (5,10,20,50),
    'clf__max_depth': (50,150,250),
    'clf__min_samples_split': (1,2,3),
    'clf__min_samples_leaf': (1,2,3)
}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,
                           verbose=1, scoring='f1')
grid_search.fit(X_train, y_train)
    # find the best configuration of hyperparameters.
print 'Best score: %.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t%s: %r' % (param_name, best_parameters[param_name])
predictions = grid_search.predict(X_test)
print classification_report(y_test, predictions)












































