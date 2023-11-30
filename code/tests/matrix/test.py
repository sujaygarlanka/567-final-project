import sys
sys.path.append('../../')
from lib.matrix_approach.layers import *
    
def test_FC():
    fc = FC(1, 2)
    fc.params = np.array([1, 2, 3, 4]).reshape(fc.params_shape)
    i = np.array([[1]])
    o = fc(i)[0]
    grad = fc.backward(np.array([[1, 2]]))
    print(o)
    print(grad)
    assert np.array_equal(o, np.array([4, 6]))
    assert np.array_equal(grad, np.array([3]))
    assert np.array_equal(fc.grads, np.array([[1, 2],[1, 2]]))
    assert np.array_equal(fc.parameters(), np.array([1, 2, 3, 4]))

def test_net():

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
    print(outputs)
#     [[0.]
#      [2.]]

    # backward
    # svm "max-margin" loss
    svm_loss = loss(outputs, o)
    print(svm_loss)
    # 0.9
    model.backward(loss)
    print(model.parameters())
    # (array([1., 1., 0.]), array([-0.05, -0.05, -0.05 ]))

def test_convolutional_layer():

    

# test_FC()
# test_net()
test_convolutional_layer()
    