import numpy as np

class Backward():

    @staticmethod
    def add(val_a, val_b, grad):
        val_a.grad += grad
        val_b.grad += grad

    @staticmethod
    def multiply(val_a, val_b, grad):
        val_a.grad += val_b.val * grad
        val_b.grad += val_a.val * grad

    @staticmethod
    def relu(val_a, grad):
        val_a.grad = val_a.grad + grad if val_a.val > 0 else val_a.grad

    @staticmethod
    def power(val_a, val_b, grad):
        grad_1 = val_b.val * val_a.val ** (val_b.val - 1) * grad
        if not np.iscomplex(grad_1):
            val_a.grad += grad_1

        grad_2 = val_a.val ** val_b.val * np.log(val_a.val) * grad
        if not np.iscomplex(grad_2):
            val_b.grad += grad_2

    @staticmethod
    def log(val_a, grad):
        val_a.grad += grad / val_a.val

class Value():

    def __init__(self, val, op=None, parents=[]):
        self.val = val
        self.op = op
        self.parents = parents
        self.grad = 0.0

    # Add
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.val + other.val, "+", [self, other])
    
    def __radd__(self, other):
        return self + other
    
    # Subtract
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)
    
    # Multiply
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.val * other.val, "*", [self, other])
    
    def __rmul__(self, other):
        return self * other
    
    # Negate
    def __neg__(self):
        return self * -1
    
    # Power
    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.val ** other.val, "**", [self, other])
    
    def __rpow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(other.val ** self.val, "**", [other, self])
    
    # Division
    def __truediv__(self, other):
        return self * other ** -1
    
    def __rtruediv__(self, other):
        return self ** -1 * other
    
    def __repr__(self):
        return str(self.val)
    
    def log(self):
        return Value(np.log(self.val), "log", [self])
    
    # Activation functions
    def relu(self):
        return Value(max(0.0, self.val), "relu", [self])
    
    def backward(self):
        topological_sort = []
        visited = set()
        # Topological sort of the graph from this node
        def dfs(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    dfs(parent)
                topological_sort.append(node)

        dfs(self)
        self.grad = 1
        for _ in range(len(topological_sort)):
            node = topological_sort.pop()
            if node.op == "+":
                Backward.add(node.parents[0], node.parents[1], node.grad)
            elif node.op == "*":
                Backward.multiply(node.parents[0], node.parents[1], node.grad)
            elif node.op == "relu":
                Backward.relu(node.parents[0], node.grad)
            elif node.op == "**":
                Backward.power(node.parents[0], node.parents[1], node.grad)
            elif node.op == "log":
                Backward.log(node.parents[0], node.grad)

