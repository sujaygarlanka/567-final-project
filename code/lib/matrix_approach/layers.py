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
        self.grads = np.empty(self.params.shape)
        self.stride = stride
        self.padding = padding
        self.params_shape = (number_filters, kernel_size, kernel_size, input_channels)
        self.trainable = True

        # Variables during forward pass to be used for the backward pass
        self.padded_input = None
        self.sub_matrices = None

    def __call__(self, inputs):
        self.padded_input = np.pad(inputs, ((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)), 'constant', constant_values=0)
        self.sub_matrices = self._get_conv_sub_matrices(self.padded_input)

        batch_size, output_height, output_width, num_filters = self._get_output_shape(self.padded_input)
        output = None 
        for p in self.params:
            weights = p[:-1]
            bias = p[-1]
            weights = weights.reshape(self.kernel_size, self.kernel_size, self.input_channels)
            convolved = np.tensordot(self.sub_matrices, weights, axes=([2, 3, 4], [0, 1, 2]))
            convolved += bias
            convolved = convolved.reshape(batch_size, output_height, output_width, 1)
            output = convolved if output is None else np.concatenate([output, convolved], axis=-1)
        return output

    def backward(self, prev_grad):
        # Calculate gradients for weights by iterating over the filters
        ###############################################################
        for i, p in enumerate(self.params):
            # Get gradients for outputs associated with the current filter
            prev_grad_filter = prev_grad[:, :, :, i]
            prev_grad_filter = prev_grad_filter.reshape(prev_grad_filter.shape[0], prev_grad_filter.shape[1] * prev_grad_filter.shape[2])
            # Get gradients for weights by multiplying the inputs by the gradients of the outputs
            weights_grads = np.einsum('ijklm,ij->ijklm', self.sub_matrices, prev_grad_filter)
            # Sum the gradients across the batch
            weights_grads = np.reshape(weights_grads, (weights_grads.shape[0]*weights_grads.shape[1], -1))
            weights_grads = np.sum(weights_grads, axis=0)
            # Get the gradient for the bias by summing the gradients of the outputs across the batch
            bias_grad = np.sum(prev_grad_filter.flatten())   
            # Update the weights and bias
            self.grads[i] = np.append(weights_grads, bias_grad)

        # Calculate gradients for inputs by iterating over the filters
        ###############################################################
        batch_size, output_height, output_width, num_filters = self._get_output_shape(self.padded_input)
        filters = self.params[:, :-1].reshape(self.params_shape)
        filter_masks = []
        
        input_num_rows = self.padded_input.shape[1]
        input_num_cols = self.padded_input.shape[2]

        for i in range(output_height):
            for j in range(output_width):
                curr_filter_mask = np.pad(filters, ((0,0),(i * self.stride, input_num_rows - (i*self.stride + self.kernel_size)),(j * self.stride, input_num_cols - (j*self.stride + self.kernel_size)),(0,0)), 'constant', constant_values=0)
                filter_masks.append(curr_filter_mask)

        # (output_width * output_height, num_filters, input_rows, input_cols, input_channels)
        filter_masks = np.array(filter_masks)
        # (1, output_width * output_height, num_filters, input_rows, input_cols, input_channels)
        filter_masks = np.expand_dims(filter_masks, axis=0)
        # (batch_size, output_width * output_height, num_filters, input_rows, input_cols, input_channels)
        filter_masks = np.repeat(filter_masks, batch_size, axis=0)
        # (batch_size, output_width * output_height, num_filters)
        prev_grad_shaped = prev_grad.reshape(prev_grad.shape[0], prev_grad.shape[1]*prev_grad.shape[2], prev_grad.shape[3])
        # (batch_size, output_width * output_height, num_filters, input_rows, input_cols, input_channels)
        # ** Need to trim if there is padding **
        backward_grads = np.einsum('ijk,ijklmn->ijklmn', prev_grad_shaped, filter_masks)
        # Sum for gradients
        return np.sum(backward_grads, axis=(1,2))

    def _get_output_shape(self, inputs):
        batch_size = inputs.shape[0]
        input_height = inputs.shape[1]
        input_width = inputs.shape[2]
        output_width = int((input_width + 2*self.padding - self.kernel_size) / self.stride + 1)
        output_height = int((input_height + 2*self.padding - self.kernel_size) / self.stride + 1)
        return batch_size, output_height, output_width, self.number_filters
    
    def _get_conv_sub_matrices(self, inputs):
        # Get the sub matrices for the convolution
        item_size = inputs.itemsize 
        batch_size = inputs.shape[0]
        input_height = inputs.shape[1]
        input_width = inputs.shape[2]
        input_channels = inputs.shape[3]

        # Get the shape of the sub matrices
        kernel_size = self.kernel_size
        __, output_height, output_width, _ = self._get_output_shape(inputs)
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

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.trainable = False
        self.grads = None

        # Variables during forward pass to be used for the backward pass
        self.input = None
        self.sub_matrices = None

    def __call__(self, input):
        self.input = input
        sub_matrices = self._get_pool_sub_matrices(self.input)
        self.sub_matrices = sub_matrices
        batch_size, output_height, output_width, output_channels = self._get_output_shape(self.input)
        # Reshape from (batch_size, num_sub_matrices, kernel_size, kernel_size, input_channels) to (batch_size * num_sub_matrices, kernel_size, kernel_size, input_channels)
        sub_matrices = np.reshape(sub_matrices, (sub_matrices.shape[0] * sub_matrices.shape[1], sub_matrices.shape[2], sub_matrices.shape[3], sub_matrices.shape[4]))
        # Get the max value for each sub matrix
        max_pool = np.max(sub_matrices, axis=(1,2))
        # Reshape from (batch_size * num_sub_matrices, input_channels) to (batch_size, output_height, output_width, output_channels)
        # Output channels is the same as input channels for max pooling
        output = np.reshape(max_pool, (batch_size, output_height, output_width, output_channels))
        return output


    def backward(self, prev_grad):
        pass


    def _get_output_shape(self, inputs):
        batch_size = inputs.shape[0]
        input_height = inputs.shape[1]
        input_width = inputs.shape[2]
        input_depth = inputs.shape[3]

        output_width = int((input_width - self.kernel_size) / self.stride + 1)
        output_height = int((input_height - self.kernel_size) / self.stride + 1)
        return batch_size, output_height, output_width, input_depth
    
    def _get_pool_sub_matrices(self, inputs):
        # Get the sub matrices for the pooling
        item_size = inputs.itemsize 
        batch_size = inputs.shape[0]
        input_height = inputs.shape[1]
        input_width = inputs.shape[2]
        input_channels = inputs.shape[3]

        # Get the shape of the sub matrices
        kernel_size = self.kernel_size
        __, output_height, output_width, _ = self._get_output_shape(inputs)
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

        
class ReLU():
        
    def __init__(self):
        self.inputs = None
        self.trainable = False

    def __call__(self, inputs):
        self.inputs = inputs
        return np.maximum(self.inputs, 0)

    def backward(self, prev_grad):
        input_grads = prev_grad * (self.inputs > 0)
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
        return input_grads
    