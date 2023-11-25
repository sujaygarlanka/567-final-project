import random
import numpy as np
# import cupy as cp
from lib.computational_graph_approach.value import Value

class Network:
    def parameters(self):
        params = np.array([])
        for layer in self.layers:
            params = np.concatenate((params, layer.parameters()))
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def save(self, path):
        np.save(path, np.array([p.val for p in self.parameters()]))

    def restore(self, path):
        saved_params = np.load(path)
        parameters = self.parameters()
        for sp, p in zip(saved_params, parameters):
            p.val = sp


class Neuron:
    def __init__(self, num_inputs, activation_function=None):
        self.num_inputs = num_inputs
        self.activation_function = activation_function
        self.weights = np.array(
            [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        )
        self.bias = Value(0.0)

    def __call__(self, inputs):
        output = 0.0
        output = np.dot(inputs, self.weights.T)
        output += self.bias

        if self.activation_function == "relu":
            return output.relu()
        else:
            return output

    def parameters(self):
        return np.concatenate((self.weights, np.array([self.bias])))

class FC:
    def __init__(self, num_inputs, num_outputs, activation_function=None):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layer = [
            Neuron(num_inputs, activation_function) for _ in range(num_outputs)
        ]

    def __call__(self, inputs):
        out = np.array([neuron(inputs) for neuron in self.layer])

        return out[0] if len(out) == 1 else out

    def parameters(self):
        params = np.array([])
        for neuron in self.layer:
            params = np.concatenate((params, neuron.parameters()))
        return params


# class Conv2D():

#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation_function=None):
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
