# Optimization

## Learning Objectives

- What is a hyperparameter?
- How and why do you normalize your input data?
- What is a saddle point?
- What is stochastic gradient descent?
- What is mini-batch gradient descent?
- What is a moving average? How do you implement it?
- What is gradient descent with momentum? How do you implement it?
- What is RMSProp? How do you implement it?
- What is Adam optimization? How do you implement it?
- What is learning rate decay? How do you implement it?
- What is batch normalization? How do you implement it?

## Tasks

| **Filename**                | **Description**                                                                                                                                                               |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0-norm_constants.py`       | Function to calculate the normalization (standardization) constants of a matrix.                                                                                              |
| `1-normalize.py`            | Function to normalize (standardizes) a matrix.                                                                                                                                |
| `2-shuffle_data.py`         | Function to shuffle the data points in two matrices the same way.                                                                                                             |
| `3-mini_batch.py`           | Function to train a loaded neural network model using mini-batch gradient descent.                                                                                            |
| `4-moving_average.py`       | Function to calculate the weighted moving average of a data set:                                                                                                              |
| `5-momentum.py`             | Function to update a variable using the gradient descent with momentum optimization algorithm.                                                                                |
| `6-momentum.py`             | Function to create the training operation for a neural network in `tensorflow` using the gradient descent with momentum optimization algorithm.                               |
| `7-RMSProp.py`              | Function to update a variable using the RMSProp optimization algorithm.                                                                                                       |
| `8-RMSProp.py`              | Function to create the training operation for a neural network in `tensorflow` using the RMSProp optimization algorithm                                                       |
| `9-Adam.py`                 | Function to update a variable in place using the Adam optimization algorithm.                                                                                                 |
| `10-Adam.py`                | Function to create the training operation for a neural network in `tensorflow` using the Adam optimization algorithm.                                                         |
| `11-learning_rate_decay.py` | Function to update the learning rate using inverse time decay in `numpy`.                                                                                                     |
| `12-learning_rate_decay.py` | Function to create a learning rate decay operation in `tensorflow` using inverse time decay.                                                                                  |
| `13-batch_norm.py`          | Function to normalize an unactivated output of a neural network using batch normalization.                                                                                    |
| `14-batch_norm.py`          | Function to creates a batch normalization layer for a neural network in `tensorflow`.                                                                                         |
| `15-model.py`               | Function to build, train, and save a neural network model in `tensorflow` using Adam optimization, mini-batch gradient descent, learning rate decay, and batch normalization. |
