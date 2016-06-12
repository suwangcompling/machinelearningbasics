### NEURAL NETS


## Basics

# types
#   i) feedforward nn
#       regular nn (acyclic) as we know it.
#   ii) feedback/recurrent nn (not implemented in scikit-learn)
#       nn that contain cycles which can represent an internal
#       state for the nn that can cause its behavior to change
#       over time based on input.

# cost function
#   mse = 1/m * \sum_i (y_i-f(x_i))^2

# parametric functions
#   i) regression function
#       y_hat = \beta_0 + \sum_i (beta_i*xi)
#   ii) activation function
#       g(x) = 1/(1+e^{-x})
#       so we compute g(y_hat)

# backward propagation (detail see A.Ng's material)
#   i) error
#       a) output-hidden
#           e_output = (true - predicted) * partial_derivative(activation)
#       b) at hidden
#           e_hidden = \sum_i (e_output_i * weight_{hidden-output_i})
#                           * partial_derivative(hidden)
#   ii) updating
#       a) input-hidden
#           += weights by learning_rate * error_hidden * val_input


## XOR Example

# preparation
#   pip install gitpython
#

from sklearn.cross_validation import train_test_split
from sklearn.neural_network import


































































