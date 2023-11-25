import sys
sys.path.append('../../')
from lib.matrix_approach.layers import *
    
def test_FC():
    fc = FC(1, 2)
    fc.params = np.array([1, 2, 3, 4]).reshape(fc.params_shape)
    o = fc(1)
    grad = fc.backward(np.array([1, 2]))
    assert np.array_equal(o, np.array([4, 6]))
    assert np.array_equal(grad, np.array([3]))
    assert np.array_equal(fc.grads, np.array([[1, 2],[1, 2]]))
    assert np.array_equal(fc.parameters(), np.array([1, 2, 3, 4]))

test_FC()
    