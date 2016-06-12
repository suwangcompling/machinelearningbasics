### NEURAL NETS


## Perceptron I: How does it work?

# components:
#   y = act_func(argument)
#   i) argument: a linear regression model
#       \sum_{i}^{p} w_i*x_i + b
#       where x_i are variables, b is intercept/bias term.
#   ii) act_func: activation function
#       e.g. 1, Heaviside Step Function
#           g(arg) = {1, if arg>0; 0, otherwise}
#       e.g. 2, Logistic Function
#           g(arg) = 1/(1+e^{arg})
#
# mechanism:
#   i) set weights to random numbers (e.g. 0)
#   ii) if estimated values produced are incorrect, update weights
#       w_i(t+1) = w_i(t) + \alpha(y_j - \hat{y_j}(t))*x_{i,j}
#       \alpha: learning rate hyperparameter.
#       y_j: true value; \hat{y_j} estimated value.
#   each pass through the training instances is called an "epoch",
#   the algorithm runs until it completes an epoch without misclassification.
#   when the algorithm doesn't converge, it returns weight values
#   after a stipulated number of runs.


## Perceptron II: Demo

# 20newsgroups data
#   classes:
#       rec.sport.hockey, rec.sport.baseball, rec.autos
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
    # import utilities
categories = ['rec.sport.hockey', 'rec.sport.baseball',
              'rec.autos']
newsgroups_train = fetch_20newsgroups(subset='train',\
                        categories=categories,\
                        remove=('headers','footers','quotes'))
newsgroups_test = fetch_20newsgroups(subset='test',\
                        categories=categories,\
                        remove=('headers','footers','quotes'))
    # load raw data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(newsgroups_train.data)
X_test = vectorizer.transform(newsgroups_test.data)
    # vectorize data
classifier = Perceptron(n_iter=100, eta0=0.1)
classifier.fit_transform(X_train,newsgroups_train.target)
predictions = classifier.predict(X_test)
print classification_report(newsgroups_test.target,predictions)


## Nonlinear I: Kernalization

# (Nonlinear II: Artificial Neural Network in next chapter)

# mechanism
#   projecting linearly inseparable data to a higher dimensional
#   space in which it is linearly separable.
#   can be used with Perceptron, but mainly used in SVM.







































