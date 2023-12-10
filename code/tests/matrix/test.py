import sys
sys.path.append('../../')
from lib.matrix_approach.layers import *
    
def test_FC():
    fc = FC(1, 2)
    fc.params = np.array([1, 2, 3, 4]).reshape(fc.params_shape)
    i = np.array([[1]])
    o = fc(i)[0]
    grad = fc.backward(np.array([[1, 2]]))
    assert np.array_equal(o, np.array([4, 6]))
    assert np.array_equal(grad, np.array([3]))
    assert np.array_equal(fc.grads, np.array([[1, 2],[1, 2]]))
    assert np.array_equal(fc.parameters(), np.array([1, 2, 3, 4]))

def test_FC_net():

    class NN(Network):
        def __init__(self):
            self.layers = [
                FC(2, 1),
                ReLU(),
            ]

            self.layers[0].params = np.array([1, 1, 0]).reshape(self.layers[0].params_shape)

        def __call__(self, input):
            return self.forward(input)
        
    # Instantiate the model
    model = NN()
    loss = SVM()

    # forward
    i = np.array([[0, 0], [1, 1]])
    o = np.array([[0], [0.1]])
    outputs = model(i)

    # svm "max-margin" loss
    svm_loss = loss(outputs, o)

    # backward
    model.backward(loss)

    assert np.array_equal(outputs, np.array([[0], [2]]))
    assert svm_loss == 0.9
    assert np.array_equal(model.parameters()[0], np.array([1, 1, 0]))
    assert np.array_equal(model.parameters()[1], np.array([-0.05, -0.05, -0.05]))

def test_convolution():
    inputs = np.array(
        [
            [
                [[1], [1], [1]],
                [[2], [2], [2]],
                [[3], [3], [3]]
            ]
        ]
    )
    conv_output = np.array(
        [
            [
                [[7], [7]],
                [[11], [11]]
            ]
        ]
    )
    layer = Conv2D(input_channels=1, kernel_size=2, number_filters=1, stride=1, padding=0)
    layer.params = np.array([[1, 1, 1, 1, 1]])
    outputs = layer(inputs)

    prev_gradients = np.array(
        [
            [
                [[1], [1]],
                [[2], [2]]
            ]
        ]
    )
    calculated_input_gradients = layer.backward(prev_gradients)
    actual_input_gradients = np.array(
        [
            [
                [[1], [2], [1]],
                [[3], [6], [3]],
                [[2], [4], [2]]
            ]
        ]
    )
    actual_gradients = np.array([[10, 10, 16, 16, 6]])


    assert np.array_equal(actual_input_gradients, calculated_input_gradients)
    assert np.array_equal(actual_gradients, layer.grads)
    assert np.array_equal(outputs, conv_output)

def test_max_pooling():
    inputs = np.array(
        [
            [
                [[1, 4], [1, 4], [1, 4], [1, 4]],
                [[2, 3], [2, 3], [2, 3], [2, 3]],
                [[3, 2], [3, 2], [3, 2], [3, 2]],
                [[4, 1], [4, 1], [4, 1], [4, 1]]
            ]
        ]
    )
    layer = MaxPool2D(kernel_size=2)
    outputs = layer(inputs)
    prev_gradients = np.array(
        [
            [
                [[1, 1], [1, 1]],
                [[2, 2], [2, 2]]
            ]
        ]
    )

    calculated_input_gradients = layer.backward(prev_gradients)
    
    assert np.array_equal(outputs, np.array([[[[2, 4], [2, 4]], [[4, 2], [4, 2]]]]))

test_FC()
test_FC_net()
test_convolution()
test_max_pooling()
    