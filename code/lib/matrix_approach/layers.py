from typing import Any
import numpy as np
import random

class Network:

    def set_parameters(self, params):
        for layer in self.layers:
            if layer.trainable:
                layer.params = params[:layer.params_shape[0] * layer.params_shape[1]].reshape(layer.params_shape)
                params = params[layer.params_shape[0] * layer.params_shape[1]:]

    def parameters(self):
        params = np.array([])
        grads = np.array([])
        for layer in self.layers:
            if layer.trainable:
                params = np.concatenate((params, layer.parameters()))
                grads = np.concatenate((grads, layer.gradients()))
        return params, grads

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def backward(self, loss_fn):
        prev_grad = loss_fn.backward(1.0)
        for layer in reversed(self.layers):
            # print(prev_grad.shape)
            prev_grad = layer.backward(prev_grad)
        return prev_grad

class FC():

    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.params_shape = (num_inputs + 1, num_outputs)
        self.grads = None
        self.params = np.array([0.1 for _ in range(self.params_shape[0] * self.params_shape[1])]).reshape(self.params_shape)
        self.inputs = None
        self.trainable = True

    def __call__(self, inputs):
        self.inputs = np.append(inputs, 1)
        return np.matmul(self.inputs, self.params)
    
    def backward(self, prev_grad):
        input_matrix = np.repeat(np.array([self.inputs]), self.params_shape[1], axis=0)
        input_matrix = np.transpose(input_matrix)
        prev_grad_2 = np.repeat(np.array([prev_grad]), self.params_shape[0], axis=0)
        self.grads = np.multiply(input_matrix, prev_grad_2)
        return np.sum(self.grads, axis=1)[:-1]
    
    def parameters(self):
        return self.params.flatten()

    def gradients(self):
        return self.grads.flatten()
        
class ReLU():
        
    def __init__(self):
        self.inputs = None
        self.trainable = False

    def __call__(self, inputs):
        self.inputs = inputs
        return np.maximum(self.inputs, 0)

    def backward(self, prev_grad):
        return prev_grad * (self.inputs >= 0)
    
class MSE():
    
    def __init__(self):
        self.inputs = None
        self.targets = None
        self.trainable = False
    
    def __call__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        return np.mean(np.square(inputs - targets))
    
    def backward(self, prev_grad):
        return 2 * prev_grad *  (self.inputs - self.targets) / len(self.inputs)
    
class SVM():
        
    def __init__(self):
        self.inputs = None
        self.targets = None
        self.trainable = False
    
    def __call__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        return np.mean(np.maximum(0, 1 - inputs * targets))
    
    def backward(self, prev_grad):
        return prev_grad * -self.targets * (self.inputs * self.targets < 1) / len(self.inputs)

    