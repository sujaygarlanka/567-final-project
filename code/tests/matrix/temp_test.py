import sys
sys.path.append('../../')

import random
import numpy as np
import matplotlib.pyplot as plt

# from IPython import embed; embed()
from sklearn.datasets import make_moons, make_blobs
from lib.matrix_approach.layers import *

np.random.seed(1337)
random.seed(1337)

# make up a dataset
X, y = make_moons(n_samples=4, noise=0.1)

y = y*2 - 1 # make y be -1 or 1
# visualize in 2D
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')

class NN(Network):

    def __init__(self):
        self.layers = [
            FC(2, 1),
            ReLU(),
            # FC(16, 16),
            # ReLU(),
            # FC(16, 1)
        ]

    def __call__(self, input):
        return self.forward(input)
    
# Instantiate the model
model = NN()
loss = SVM()

Xb, yb = X, y

for k in range(100):
    
    # forward
    scores = model(Xb)

    # backward
    # svm "max-margin" loss
    svm_loss = loss(scores, np.expand_dims(yb, axis=1))
    model.backward(loss)

    # apply l2 regularization
    # l2_loss = model.l2_regularization()
    # total_loss = svm_loss + l2_loss
    total_loss = svm_loss
    # from IPython import embed; embed()
    accuracy = [(yi > 0) == (scorei[0] > 0) for yi, scorei in zip(yb, scores)]
    acc = sum(accuracy) / len(accuracy)
    
    # update (sgd)
    # learning_rate = 1.0 - 0.9*k/100
    learning_rate = 0.001
    params, grads = model.parameters()
    updated_params = params + -1 * learning_rate * grads
    model.set_parameters(updated_params)
    
    if k % 1 == 0:
        print(f"step {k} loss {total_loss}, accuracy {acc*100}%")

print(scores)
print(yb)