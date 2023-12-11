# 567-final-project

## Contents
- code
    - cartpole: code related to training and running agent for the Cartpole V1 environment
    - flappy_bird: code related to train and running agent for the Flappy Bird environment
    - lib: neural network libraries
        - computational_graph_approach
        - matrix_approach (layered implementation)
    - tests: testing neural network libraries
        - computational_graph
        - matrix (layered approach tests)

## Implement Deep Q Network From Scratch To Train An Agent

### Cartpole V1

Agents are trained to solve the cartpole environment using the computational graph approach and layered approach. The links to the jupyter notebook for both are below. Running them, you can see the different in speed between the two implementations.

- [Computational Graph](./code/cartpole/cartpole_computational_graph.ipynb)
- [Layered Approach Graph](./code/cartpole/cartpole_matrix.ipynb)

### Flappy Bird (Non RGB)
![](flappy_bird_demo.gif)

Agent is trained to play Flappy Bird using state information about the environment including the x and y position difference between the bird and the next pipe.

The training code can be found [here](./code/flappy_bird/flappy.ipynb).

To run the agent for this environemnt, simply run  [run_agent.py](./code/flappy_bird/run_agent.py) 

### Note:
To run the agent, you must have a python 3 version before 3.9 and install the following in your environment:

```
flappy-bird-gym
setuptools==65.5.0
wheel==0.38.4
```

These requirements are necessary to install the following flappy bird [environment](https://github.com/Talendar/flappy-bird-gym). Tried making it compatible with the more modern flappy bird gymnasium [environment](https://github.com/markub3327/flappy-bird-gymnasium/tree/main), but this seems to have issues with its state information. Because of this, running the evironment to test the agent works on everything except ARM macs.

### Flappy Bird (RGB)

Agent is attempted to be trained to play Flappy Bird using just RGB images. The training code for this can be found [here](./code/flappy_bird/flappy_bird_rgb.ipynb). The issues encountered is training is very slow and the loss is exploding. Input is appropriately preprocessed and normalized between 0 and 1 and L2 regularization is applied. Additional tuning needs to be done, including possibly gradient clipping. Also, trying to run this on a GPU by using cupy as a drop in replacement for numpy may have some speed ups. Overall, the convolutional layers work as well as the training loop. Tests for the layers can be found in the `tests` folder and the training loop worked on training the non RGB agent.

