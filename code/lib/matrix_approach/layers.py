from typing import Any
import numpy as np
import random

class Network:

    def set_parameters(self, params):
        for layer in self.layers:
            if layer.trainable:
                layer.params = params[:layer.params_shape[0] * layer.params_shape[1]].reshape(layer.params_shape)
                params = params[layer.params_shape[0] * layer.params_shape[1]:]

    def set_gradients(self, grads):
        for layer in self.layers:
            if layer.trainable:
                layer.grads = grads[:layer.params_shape[0] * layer.params_shape[1]].reshape(layer.params_shape)
                grads = grads[layer.params_shape[0] * layer.params_shape[1]:]

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
            prev_grad = layer.backward(prev_grad)
        return prev_grad
    
    def l2_regularization(self):
        alpha = 1e-4
        params, grads = self.parameters()
        grads += alpha * 2 * params
        self.set_gradients(grads)
        return np.sum(alpha * params * params)

class FC():

    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.params_shape = (num_inputs + 1, num_outputs)
        self.grads = None
        self.params = np.array([random.uniform(-1, 1) for _ in range(self.params_shape[0] * self.params_shape[1])]).reshape(self.params_shape)
        self.inputs = None
        self.trainable = True

    def __call__(self, inputs):
        self.inputs = np.column_stack((inputs, np.ones(len(inputs))))
        return np.matmul(self.inputs, self.params)
    
    def backward(self, prev_grad):    
        # Calculate gradients for weights
        # 3D matrix of shape (batch_size, num_outputs, num_inputs + 1)
        input_matrix = np.expand_dims(self.inputs, axis=1)
        input_matrix = np.repeat(input_matrix, self.params_shape[1], axis=1)

        # 3D matrix of shape (batch_size, num_outputs, num_inputs + 1)
        prev_grad_matrix = np.expand_dims(prev_grad, axis=2)
        prev_grad_matrix = np.repeat(prev_grad_matrix, self.params_shape[0], axis=2)

        # 3D matrix of shape (batch_size, num_outputs, num_inputs + 1)
        grad_per_input = np.multiply(input_matrix, prev_grad_matrix)
        self.grads = np.transpose(np.sum(grad_per_input, axis=0))

        # Calculate gradients for inputs
        params_without_bias = self.params[:-1]
        input_grads = []
        for i in range(len(self.inputs)):
            prev_grad[i]
            input_grads.append(np.transpose(np.matmul(params_without_bias, np.transpose(prev_grad[i]))))
        return np.array(input_grads)

    def parameters(self):
        return self.params.flatten()

    def gradients(self):
        return self.grads.flatten()
    
class Conv2D():
    def __init__(self, input_channels, kernel_size, number_filters, 
                stride=1, padding=0):
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.params = np.array([random.uniform(-1, 1) for _ in range(number_filters * (kernel_size * kernel_size * input_channels + 1))]).reshape(number_filters, -1)
        self.stride = stride
        self.padding = padding
        self.params_shape = (number_filters, kernel_size, kernel_size, input_channels)
        self.padded_input = None

    def _get_output_shape(self, inputs):
        batch_size = inputs.shape[0]
        input_height = inputs.shape[1]
        input_width = inputs.shape[2]
        output_width = int((input_width + 2*self.padding - self.kernel_size) / self.stride + 1)
        output_height = int((input_height + 2*self.padding - self.kernel_size) / self.stride + 1)
        return batch_size, output_height, output_width, self.number_filters

    def forward(self, inputs):
        self.padded_input = np.pad(inputs, ((0,0),(1,1),(1,1),(0,0)), 'constant', constant_values=0)
        sub_matrices = self._get_conv_sub_matrices(self.padded_input)
        
        batch_size, output_height, output_width, num_filters = self._get_output_shape(inputs)
        output = np.empty((batch_size, output_height, output_width, num_filters), dtype=inputs.dtype)  
        for p in self.params:
            weights = p[:-1]
            bias = p[-1]
            weights = weights.reshape(self.kernel_size, self.kernel_size, self.input_channels)
            convolved = np.tensordot(sub_matrices, weights, axes=([2, 3, 4], [0, 1, 2]))
            convolved += bias
            convolved = convolved.reshape(batch_size, output_height, output_width, 1)
            output = np.dstack((output, convolved))

        return output

    def backward(self, prev_grad):
        sub_matrices = self._get_conv_sub_matrices(self.padded_input)
        # Calculate gradients for weights
        for i, p in enumerate(self.params):
            prev_grad_filter = prev_grad[:, :, :, i].squeeze()
            grads = np.einsum('ijkl,ijk->ijkl', sub_matrices, prev_grad_filter)
            


    def _get_conv_sub_matrices(self, inputs):
        # Get the sub matrices for the convolution
        item_size = inputs.item_size 
        batch_size = inputs.shape[0]
        input_height = inputs.shape[1]
        input_width = inputs.shape[2]
        input_channels = inputs.shape[3]

        # 1. Get the shape of the sub matrices
        kernel_size = self.kernel_size
        __, output_height, output_width, _ = self.get_output_shape(inputs)

        shape = (batch_size, output_height, output_width, kernel_size, kernel_size, input_channels)

        # Get strides of the sub matrices
        channel = item_size
        column = channel * input_channels
        row = column * input_width
        horizontal_stride = self.stride * column
        vertical_stride = self.stride * row
        batch = input_height * input_width * column
        stride_shape = (batch, vertical_stride, horizontal_stride, row, column, channel)

        # Get the sub matrices
        sub_matrices = np.lib.stride_tricks.as_strided(inputs, shape=shape, strides=stride_shape)

        # Combine output height and output width
        sub_matrices = np.reshape(sub_matrices, (batch_size, output_height * output_width, kernel_size, kernel_size, input_channels))

        return sub_matrices
    

class MaxPool2D():

    def __init__(self, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.trainable = False


        
class ReLU():
        
    def __init__(self):
        self.inputs = None
        self.trainable = False

    def __call__(self, inputs):
        self.inputs = inputs
        return np.maximum(self.inputs, 0)

    def backward(self, prev_grad):
        input_grads = prev_grad * (self.inputs > 0)
        # return np.sum(grads, axis=0)
        return input_grads
    
class MSE():
    
    def __init__(self):
        self.inputs = None
        self.targets = None
        self.trainable = False
    
    def __call__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        return np.mean(np.square(inputs - targets), axis=1)
    
    def backward(self, prev_grad):
        return 2 * prev_grad * (self.inputs - self.targets) / self.inputs.shape[1]
    
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
        input_grads = prev_grad * -self.targets * (self.inputs * self.targets < 1) / len(self.inputs)
        # print(np.sum(grads, axis=0))
        # return np.sum(grads, axis=0)
        return input_grads
    