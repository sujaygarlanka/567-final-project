
# Implement Neural Network Library and Train RL Agent

My goal was to implement an efficient neural network library from scratch and use it to create a Deep Q Network (DQN) RL agent to play Flappy Bird.

<img style="max-width: 300px" width="100%" src="https://raw.githubusercontent.com/sujaygarlanka/567-final-project/main/flappy_bird_demo.gif"/>

## Neural Network Library

The are many approaches to developing a neural network library, I investigated two approaches, the computational graph approach and the layered approach. Both have their strengths and weaknesses. I implemented both and benchmarked them by training a DQN to successfully solve the Cart Pole problem.

### Computational Graph

The computational graph approach, as the name suggests, is building a graph that represents all the computations taking place in a neural network, then implementing backpropagation over the graph to compute the gradients for all values involved in the computations.

My implementation is a reimplementation of Andrej Karpathy's micrograd with some syntatic modifications. For a very thorough explanation you can refer to his video. Below, I summarize the implementation details.

#### Graph

For the computational graph an essential component is representing a computation. A computation is simply a mathematical operation between two values. In my case, these values are all floats. To represent these computations, I wrote a Value class in python to wrap values. The Value class contains a property for the float value, but also stores the operation between the two values that produced the float value. With this, it stores half the information about the computation. The other half is the values that are operated on for the computation. In addition, references to the Values classes of the two values that produced the float value are stored in the Value class. So, the Value class ultimately contains a float value, an operation, and the two Values that produced the float value as shown in the table below. With this, every Value encapsulates a computation. The way this computation fits in a graph is shown in the image below where the blue rectangle is the single computation or single instantiation of the Value class.

| Property | Description                                                                                      |
|----------|--------------------------------------------------------------------------------------------------|
| val      | A float value                                                                                    |
| op       | A string describing the operation executed to produce the value                                  |
| parents  | The references to the two values that the operation was applied between to produce the float value|
| grad     | The gradient for this value                                                                      |

To create the computational graph to represent a series of computations, I overwrote the basic operators in the Value class. The overwritten operators are add, subtract, multiply, divide, negation and exponentiation. I overwrote these operators, so when a Value is operated on with the operators listed above with another Value, the output is a new Value that points to these two parent Values. Repeatedly doing this will create a graph of Values where each Value is the child of the two parent Values that were involved in a computation.

The code for all of the Value class is little over a 100 lines and is below and all the related code can be found [here](https://github.com/sujaygarlanka/567-final-project/tree/main/code/lib/computational_graph_approach).

```python
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

```

#### Backpropagation

The last feature to implement is backpropagation. It is the method to compute the gradients for each Value in the computational graph. Backpropagation is implemented by iterating from the final output Value to the inputs. While iterating, each Value computes the gradients for its parents. The child Value has the information of the parent Values, the operation applied between the Values, and its own gradient. It uses this information to find the gradient of each parent with respect to the operation and the Value of the other parent and multiplies it with its own gradient. This produces the gradient for each parent with respect to the one computation that produced this child Value. However, a problem remains because the parents may be involved in multiple computations. To solve this issue, the calculated gradient for each parent is simply added to its current gradient. This ensures that for every computation a Value is involved in, its gradient is not overwritten, but accumulated to find the total change (gradient) a Value can affect on the neural network output.

When implementing backpropagation as described above in the computational graph, the graph must be traversed in a specific order. Since the gradient of each Value depends on the gradient of its child Values, the gradient of the child Values must be completely calculated before moving to the parent. This is only possible in directed acyclic graphs, which neural networks are. To traverse the graph with the restriction imposed above, a topological sort of the directed acyclic computational graph must be found. In ,y implementation, this is done via a recursive method. I traverse over the nodes and compute the gradient in the order returned by sort. How a topological sort works is shown in the diagram below.

### Layered Approach

The layered approach is the second approach I undertook. I did this with the expectation that this design would allow me to write the library with matrix operations to allow for parallelization and the utilization of efficient computational libraries such as numpy and cudapy. Numpy scientific computing library that runs the matrix operations efficiently on the CPU. CudaPy is an implementation of numpy that runs on the GPU.

My layered approach consists of a separate class for each type of layer. This varies from the computational graph approach in that each layer consists of a forward pass function to compute the output of the layer and a backward pass to compute the gradients. The computational graph had all functionality (i.e. forward pass and backward pass) at the individual value level rather than at the layer level. Doing this pass at the layer level allows us to compute all outputs, weight gradients and input gradients for each layer via matrix operations. This approach additionally allows for efficiency gains across another dimension, the batch size. The computational graph required passing in each input independently through the network. However, the layered approach allows for the all calculations to be done in parallel across all inputs that make up a batch.

The basic architecture of each layer is a forward pass method that takes in a batch of inputs and produces a batch of outputs, with each entry in the output batch corresponding to its entry in the inputs. The backward pass does two important calculations. The first is computing the gradients with respect to all the parameters (weights and biases) in the layer. The second is computing the gradients with respect to all the inputs. The input gradients are then returned by the backward pass and passed to the previous layer, so the previous layer has the gradients with respect to its outputs (outputs of previous layer are inputs of current layer). With this architecture, the backward pass is simply iterating over each layer, calling the backward method, getting the output and passing it to the next layer.
The full implementation can be found here.

The type of layers I implemented in this approach are fully connected, convolution layer, mean squared error and flatten. The implementation details for most of these layers are standard except for convolution and max pooling. These two layers have windowing, which is traditionally a serially implemented with for loops. However, the main benefit of this approach is parallelization via matrix operations, so I aimed to find a matrix approach. Luckily, both numpy and cudapy have an as strided function that allows for parallelized windowing by accessing the underlying bytes.


### Comparison

I used both implementations to train a DQN to solve the Cart Pole problem. The results table below show that layered approach running on the CPU is fastest by 100x. It is faster than the computational graph approach as expected, however it is unexpectedly slower than the layered approach on the GPU. I suspect it may be due to the time it takes to move the data from the CPU to the GPU the number of times required for training the RL agent.

| Method                    | Forward Pass (ms) | Backward Pass (ms) | Total Time - 1000 iterations (s) |
|---------------------------|-------------------|--------------------|----------------------------------|
| Computational Graph       | 2.47              | 0.66               | 99.95                            |
| Layered (numpy/CPU)       | 0.011             | 0.048              | 1.57                             |
| Layered (cupy/GPU)        | 0.31              | 0.57               | 28.35                            |

## RL Agent
I built an RL agent to play Flappy Bird. Flappy Bird is a game that requires the player to have a bird looking sprite navigate through gaps in pipes. The sprite has a constant x velocity as it moves towards the pipes. The player has the ability to apply a force in the y direction. The goal is for the player to time the application of the force to navigate through the gaps between the pipes for as long as possible.

### Environment

For Flappy Bird, I used an environment following the gym API to train and evaluate my agent.

**Observation Space**
- x difference between the bird and the gap in the next pipe
- y difference between the bird and the gap in the next pipe

**Action Space**
- 0 applying no force and 1 applying a set force.

**Reward**
- A value of 1 returned every time step the bird did not crash.

**Termination Condition**
- Crashing into a pipe or ground or going above the top of the screen

 One thing worth pointing out is that the Markov assumption does not fully hold for the state because the velocity of the bird is not included. I only have positional information. However, this proves to be sufficient, but may reduce the efficieny of the training.

### DQN Agent

#### DQN Algorithm
The DQN agent uses a Deep Q Network (DQN) to learn a policy. To understand a deep Q network, I will explains its foundation, which is Q-learning. Q-learning works with the assumption that states/observations follow the Markov property. This means that the state encapsulates all information about the environment that allows for the agent to find an optimal action. It is centered around the Q function. The Q function takes in the state and action and produces a value. The higher the values, the better the state action pair. In Q-learning, the goal is to learn this Q function. From this Q function, an agent can find the optimal action when given a state by finding the action that returns the highest Q value when paired with the state. The Q function is ultimately learned via repeated Bellman updates to the Q function. As the equation below shows, the Q value for the current state-action pair, is the max Q value for the next state multiplied by discount factor (γ), added to the reward that was returned for getting to the next state. Repeatedly running this update with collected returns an optimal Q function. For a DQN, this Q function is represented by a neural network and is updated via a gradient step. This gradient step is taken at the end of each episode on a batch of experiences saved from previous episodes. This buffer of saved experiences is called replay memory. The full algorithm for a DQN is described in figure 1.

#### Neural Network and Training Parameters

**Neural Network Architecture:**
Used the following neural network architecture below with a mean squared loss function, the epsilon decay rate function below and a sampling of random actions to weighted 75 percent towards applying no force and 25 percent towards applying a force. This allowed for more efficient exploration of the space because apply a force as often as not applying one quickly results in crashes in the beginning.

1. Fully Connected (in: 2, out: 64)
2. ReLU
3. Fully Connected (in: 64, out: 128)
4. ReLU
5. Fully Connected (in: 128, out: 2)

**Training Parameters**
- Discount factor (γ): 0.95
- Batch size: 32
- Leanning rate: 0.001
- Replay memory size: 2000

### Results

After training for 576,000 episodes (2 hours on a personal PC), the network converged at a solu- tion for an agent that averaged over 2000 time steps when playing Flappy Bird. This results in an average score of navigating through 30 pipes. While this is not super human performance, it shows that the agent can play the game. With more training time, the agent would be able to play at super human levels. The graph below (figure 4) plots the average episode length of running the agent for 10 episodes after every 1000 episodes of training.
