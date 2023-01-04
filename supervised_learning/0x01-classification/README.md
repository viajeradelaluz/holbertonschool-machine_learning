# Classification

## Background Context

At the end of this project, you should be able to build your own binary image classifier from scratch using numpy. Good luck and have fun!

## Learning Objectives

- What is a model?
- What is supervised learning?
- What is a prediction?
- What is a node?
- What is a weight?
- What is a bias?
- What are activation functions?\* Sigmoid?
- Tanh?
- Relu?
- Softmax?
- What is a layer?
- What is a hidden layer?
- What is Logistic Regression?
- What is a loss function?
- What is a cost function?
- What is forward propagation?
- What is Gradient Descent?
- What is back propagation?
- What is a Computation Graph?
- How to initialize weights/biases
- The importance of vectorization
- How to split up your data
- What is multiclass classification?
- What is a one-hot vector?
- How to encode/decode one-hot vectors
- What is the softmax function and when do you use it?
- What is cross-entropy loss?
- What is pickling in Python?

## More Info

### Matrix Multiplications

For all matrix multiplications in the following tasks, please use [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)

### Testing your code

In order to test your code, you’ll need DATA! Please download these datasets ([Binary_Train.npz](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-ml/Binary_Train.npz), [Binary_Dev.npz](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-ml/Binary_Dev.npz), [MNIST.npz](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-ml/MNIST.npz)) to go along with all of the following main files. You do not need to upload these files to GitHub. Your code will not necessarily be tested with these datasets. All of the following code assumes that you have stored all of your datasets in a separate `data` directory.

## Tasks

### 0.Neuron

Write a class `Neuron` that defines a single neuron performing binary classification:

- Main file: `0-main.py`
- class constructor: `def __init__(self, nx)`:
  - `nx` is the number of input features to the neuron
    - If `nx` is not an integer, raise a `TypeError` with the exception: `nx must be an integer`
    - If `nx` is less than 1, raise a `ValueError` with the exception: `nx must be a positive integer`
  - All exceptions should be raised in the order listed above
- Public instance attributes:
  - `W`: The weights vector for the neuron. Upon instantiation, it should be initialized using a random normal distribution.
  - `b`: The bias for the neuron. Upon instantiation, it should be initialized to 0.
  - `A`: The activated output of the neuron (prediction). Upon instantiation, it should be initialized to 0.

```bash
alexa@ubuntu-xenial:$ ./0-main.py
[[ 1.76405235e+00  4.00157208e-01  9.78737984e-01  2.24089320e+00
   1.86755799e+00 -9.77277880e-01  9.50088418e-01 -1.51357208e-01

...

  -5.85865511e-02 -3.17543094e-01 -1.63242330e+00 -6.71341546e-02
   1.48935596e+00  5.21303748e-01  6.11927193e-01 -1.34149673e+00]]
(1, 784)
0
0
10
alexa@ubuntu-xenial:$
```

### 1. Privatize Neuron

Write a class `Neuron` that defines a single neuron performing binary classification (Based on `0-neuron.py`):

- Main file `1-main.py`
- class constructor: `def __init__(self, nx)`:
  - `nx` is the number of input features to the neuron
    - If `nx` is not an integer, raise a `TypeError` with the exception: `nx must be a integer`
    - If `nx` is less than 1, raise a `ValueError` with the exception: `nx must be positive`
  - All exceptions should be raised in the order listed above
- **Private** instance attributes:
  - `__W`: The weights vector for the neuron. Upon instantiation, it should be initialized using a random normal distribution.
  - `__b`: The bias for the neuron. Upon instantiation, it should be initialized to 0.
  - `__A`: The activated output of the neuron (prediction). Upon instantiation, it should be initialized to 0.
    Each private attribute should have a corresponding getter function (no setter function).

```bash
alexa@ubuntu-xenial:$ ./1-main.py
[[ 1.76405235e+00  4.00157208e-01  9.78737984e-01  2.24089320e+00
   1.86755799e+00 -9.77277880e-01  9.50088418e-01 -1.51357208e-01

...

  -5.85865511e-02 -3.17543094e-01 -1.63242330e+00 -6.71341546e-02
   1.48935596e+00  5.21303748e-01  6.11927193e-01 -1.34149673e+00]]
0
0
Traceback (most recent call last):
  File "./1-main.py", line 16, in <module>
    neuron.A = 10
AttributeError: can't set attribute
alexa@ubuntu-xenial:$
```

### 2. Neuron Forward Propagation

- Write a class `Neuron` that defines a single neuron performing binary classification (Based on `1-neuron.py`):

- Main file: `2-main.py`
- Add the public method `def forward_prop(self, X)`:
  - Calculates the forward propagation of the neuron
  - `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
    - `nx` is the number of input features to the neuron
    - `m` is the number of examples
  - Updates the private attribute `__A`
  - The neuron should use a sigmoid activation function
  - Returns the private attribute `__A`

```bash
alexa@ubuntu-xenial:$ ./2-main.py
[[5.34775247e-10 7.24627778e-04 4.52416436e-07 ... 8.75691930e-05
  1.13141966e-06 6.55799932e-01]]
alexa@ubuntu-xenial:$
```

### 3. Neuron Cost

Write a class Neuron that defines a single neuron performing binary classification (Based on `2-neuron.py`):

- Main file: `3-main.py`
- Add the public method `def cost(self, Y, A)`:
  - Calculates the cost of the model using logistic regression
  - `Y` is a `numpy.ndarray` with shape (1, `m`) that contains the correct labels for the input data
  - `A` is a `numpy.ndarray` with shape (1, `m`) containing the activated output of the neuron for each example
  - To avoid division by zero errors, please use `1.0000001 - A` instead of `1 - A`
  - Returns the cost

```bash
alexa@ubuntu-xenial:$ ./3-main.py
4.365104944262272
alexa@ubuntu-xenial:$
```

### 4. Evaluate Neuron

Write a class `Neuron` that defines a single neuron performing binary classification (Based on `3-neuron.py`):

- Main file: `4-main.py`
- Add the public method `def evaluate(self, X, Y)`:
  - Evaluates the neuron’s predictions
  - `X` is a `numpy.ndarray` with shape (nx, `m`) that contains the input data
    - `nx` is the number of input features to the neuron
    - `m` is the number of examples
  - `Y` is a `numpy.ndarray` with shape (1, `m`) that contains the correct labels for the input data
  - Returns the neuron’s prediction and the cost of the network, respectively
    - The prediction should be a `numpy.ndarray` with shape (1, `m`) containing the predicted labels for each example
    - The label values should be 1 if the output of the network is >= 0.5 and 0 otherwise

```bash
alexa@ubuntu-xenial:$ ./4-main.py
[[0 0 0 ... 0 0 0]]
4.365104944262272
alexa@ubuntu-xenial:$
```

### 5. Neuron Gradient Descent

Write a class Neuron that defines a single neuron performing binary classification (Based on `4-neuron.py`):

- Main file: `5-main.py`
- Add the public method `def gradient_descent(self, X, Y, A, alpha=0.05)`:
  - Calculates one pass of gradient descent on the neuron
  - `X` is a `numpy.ndarray` with shape (nx, `m`) that contains the input data
    - `nx` is the number of input features to the neuron
    - `m` is the number of examples
  - `Y` is a `numpy.ndarray` with shape (1, `m`) that contains the correct labels for the input data
  - `A` is a `numpy.ndarray` with shape (1, `m`) containing the activated output of the neuron for each example
  - `alpha` is the learning rate
  - Updates the private attributes `__W` and `__b`

```bash
alexa@ubuntu-xenial:$ ./5-main.py
[[ 1.76405235e+00  4.00157208e-01  9.78737984e-01  2.24089320e+00
   1.86755799e+00 -9.77277880e-01  9.50088418e-01 -1.51357208e-01

...

  -5.85865511e-02 -3.17543094e-01 -1.63242330e+00 -6.71341546e-02
   1.48935596e+00  5.21303748e-01  6.11927193e-01 -1.34149673e+00]]
0.2579495783615682
alexa@ubuntu-xenial:$
```

### 6. Train Neuron

Write a class `Neuron` that defines a single neuron performing binary classification (Based on `5-neuron.py`):

- Main file: `6-main.py`
- Add the public method `def train(self, X, Y, iterations=5000, alpha=0.05)`:
  - Trains the neuron
    - `X` is a `numpy.ndarray` with shape (nx, `m`) that contains the input data
      - `nx` is the number of input features to the neuron
      - `m` is the number of examples
    - `Y` is a `numpy.ndarray` with shape (1, `m`) that contains the correct labels for the input data
    - `iterations` is the number of iterations to train over
      - if `iterations` is not an integer, raise a `TypeError` with the exception `iterations must be an integer`
      - if `iterations` is not positive, raise a `ValueError` with the exception `iterations must be a positive integer`
    - `alpha` is the learning rate
      - if `alpha` is not a float, raise a `TypeError` with the exception `alpha must be a float`
      - if `alpha` is not positive, raise a `ValueError` with the exception `alpha must be positive`
    - All exceptions should be raised in the order listed above
    - Updates the private attributes `__W`, `__b`, and `__A`
    - You are allowed to use one loop
    - Returns the evaluation of the training data after `iterations` of training have occurred

```bash
alexa@ubuntu-xenial:$ ./6-main.py
Train cost: 1.3805076999
Train accuracy: 64.737465456%
Train data: [[0 0 0 ... 0 0 1]]
Train Neuron A: [[2.70000000e-08 2.18229559e-01 1.63492900e-04 ... 4.66530830e-03
  6.06518000e-05 9.73817942e-01]]
Dev cost: 1.4096194345
Dev accuracy: 64.4917257683%
Dev data: [[1 0 0 ... 0 0 1]]
Dev Neuron A: [[0.85021134 0.         0.3526692  ... 0.10140937 0.         0.99555018]]
```

### 7. Upgrade Train Neuron

Write a class Neuron that defines a single neuron performing binary classification (Based on `6-neuron.py`):

- Main file: `7-main.py`
- Update the public method train to `def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100)`:
  - Trains the neuron by updating the private attributes `__W`, `__b`, and `__A`
  - `X` is a `numpy.ndarray` with shape (nx, `m`) that contains the input data
    - `nx` is the number of input features to the neuron
    - `m` is the number of examples
  - `Y` is a `numpy.ndarray` with shape (1, `m`) that contains the correct labels for the input data
  - `iterations` is the number of iterations to train over
    - if `iterations` is not an integer, raise a `TypeError` with the exception `iterations must be an integer`
    - if `iterations` is not positive, raise a `ValueError` with the exception `iterations must be a positive integer`
  - `alpha` is the learning rate
    - if `alpha` is not a float, raise a `TypeError` with the exception `alpha must be a float`
    - if `alpha` is not positive, raise a `ValueError` with the exception `alpha must be positive`
  - `verbose` is a boolean that defines whether or not to print information about the training. If `True`, print `Cost after {iteration} iterations: {cost}` every `step` iterations:
    - Include data from the 0th and last iteration
  - `graph` is a boolean that defines whether or not to graph information about the training once the training has completed. If `True`:
    - Plot the training data every `step` iterations as a blue line
    - Label the x-axis as `iteration`
    - Label the y-axis as `cost`
    - Title the plot `Training Cost`
    - Include data from the 0th and last iteration
  - Only if either `verbose` or `graph` are `True`:
    - if `step` is not an integer, raise a `TypeError` with the exception `step must be an integer`
    - if `step` is not positive or is greater than iterations, raise a `ValueError` with the exception `step must be positive and <= iterations`
  - All exceptions should be raised in the order listed above
  - The 0th iteration should represent the state of the neuron before any training has occurred
  - You are allowed to use one loop
  - You can use `import matplotlib.pyplot as plt`
  - Returns: the evaluation of the training data after `iterations` of training have occurred

```bash
alexa@ubuntu-xenial:$ ./7-main.py
Cost after 0 iterations: 4.365104944262272
Cost after 100 iterations: 0.11955134491351888

...

Cost after 3000 iterations: 0.013386353289868338

...

Train cost: 0.013386353289868338
Train accuracy: 99.66837741808132%
Dev cost: 0.010803484515167197
Dev accuracy: 99.81087470449172%
```

### 8. NeuralNetwork

Write a class `NeuralNetwork` that defines a neural network with one hidden layer performing binary classification:

- Main file: `8-main.py`
- class constructor: `def __init__(self, nx, nodes)`:
  - `nx` is the number of input features
    - If `nx` is not an integer, raise a TypeError with the exception: `nx must be an integer`
    - If `nx` is less than 1, raise a ValueError with the exception: `nx must be a positive integer`
  - `nodes` is the number of nodes found in the hidden layer
    - If `nodes` is not an integer, raise a `TypeError` with the exception: `nodes must be an integer`
    - If `nodes` is less than 1, raise a `ValueError` with the exception: `nodes must be a positive integer`
  - All exceptions should be raised in the order listed above
- Public instance attributes:
  - `W1`: The weights vector for the hidden layer. Upon instantiation, it should be initialized using a random normal distribution.
  - `b1`: The bias for the hidden layer. Upon instantiation, it should be initialized with 0’s.
  - `A1`: The activated output for the hidden layer. Upon instantiation, it should be initialized to 0.
  - `W2`: The weights vector for the output neuron. Upon instantiation, it should be initialized using a random normal distribution.
  - `b2`: The bias for the output neuron. Upon instantiation, it should be initialized to 0.
  - `A2`: The activated output for the output neuron (prediction). Upon instantiation, it should be initialized to 0.

```bash
alexa@ubuntu-xenial:$ ./8-main.py
[[ 1.76405235  0.40015721  0.97873798 ...  0.52130375  0.61192719
  -1.34149673]
 [ 0.47689837  0.14844958  0.52904524 ...  0.0960042  -0.0451133
   0.07912172]
 [ 0.85053068 -0.83912419 -1.01177408 ... -0.07223876  0.31112445
  -1.07836109]]
(3, 784)
[[0.]
 [0.]
 [0.]]
[[ 1.06160017 -1.18488744 -1.80525169]]
(1, 3)
0
0
0
10
alexa@ubuntu-xenial:$
```

### 9. Privatize NeuralNetwork

Write a class `NeuralNetwork` that defines a neural network with one hidden layer performing binary classification (based on `8-neural_network.py`):

- Main file: `9-main.py`
- class constructor: `def __init__(self, nx, nodes)`:
  - `nx` is the number of input features
    - If `nx` is not an integer, raise a TypeError with the exception: `nx must be an integer`
    - If `nx` is less than 1, raise a ValueError with the exception: `nx must be a positive integer`
  - `nodes` is the number of nodes found in the hidden layer
    - If `nodes` is not an integer, raise a `TypeError` with the exception: `nodes must be an integer`
    - If `nodes` is less than 1, raise a `ValueError` with the exception: `nodes must be a positive integer`
  - All exceptions should be raised in the order listed above
- **Private** instance attributes:
  - `W1`: The weights vector for the hidden layer. Upon instantiation, it should be initialized using a random normal distribution.
  - `b1`: The bias for the hidden layer. Upon instantiation, it should be initialized with 0’s.
  - `A1`: The activated output for the hidden layer. Upon instantiation, it should be initialized to 0.
  - `W2`: The weights vector for the output neuron. Upon instantiation, it should be initialized using a random normal distribution.
  - `b2`: The bias for the output neuron. Upon instantiation, it should be initialized to 0.
  - `A2`: The activated output for the output neuron (prediction). Upon instantiation, it should be initialized to 0.
  - Each private attribute should have a corresponding getter function (no setter function).

```bash
alexa@ubuntu-xenial:$ ./9-main.py
[[ 1.76405235  0.40015721  0.97873798 ...  0.52130375  0.61192719
  -1.34149673]
 [ 0.47689837  0.14844958  0.52904524 ...  0.0960042  -0.0451133
   0.07912172]
 [ 0.85053068 -0.83912419 -1.01177408 ... -0.07223876  0.31112445
  -1.07836109]]
[[0.]
 [0.]
 [0.]]
[[ 1.06160017 -1.18488744 -1.80525169]]
0
0
0
Traceback (most recent call last):
  File "./9-main.py", line 19, in <module>
    nn.A1 = 10
AttributeError: can't set attribute
alexa@ubuntu-xenial:$
```

### 10. NeuralNetwork Forward Propagation

Write a class `NeuralNetwork` that defines a neural network with one hidden layer performing binary classification (based on `9-neural_network.py`):

- Main file: `10-main.py`
- Add the public method `def forward_prop(self, X)`:
  - Calculates the forward propagation of the neural network
  - `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
    - `nx` is the number of input features to the neuron
    - `m` is the number of examples
  - Updates the private attributes `__A1` and `__A2`
  - The neurons should use a sigmoid activation function
  - Returns the private attributes `__A1` and `__A2`, respectively

```bash
alexa@ubuntu-xenial:$ ./10-main.py
[[5.34775247e-10 7.24627778e-04 4.52416436e-07 ... 8.75691930e-05
  1.13141966e-06 6.55799932e-01]
 [9.99652394e-01 9.99999995e-01 6.77919152e-01 ... 1.00000000e+00
  9.99662771e-01 9.99990554e-01]
 [5.57969669e-01 2.51645047e-02 4.04250047e-04 ... 1.57024117e-01
  9.97325173e-01 7.41310459e-02]]
[[0.23294587 0.44286405 0.54884691 ... 0.38502756 0.12079644 0.593269  ]]
alexa@ubuntu-xenial:$
```

### 11. NeuralNetwork Cost

Write a class `NeuralNetwork` that defines a neural network with one hidden layer performing binary classification (based on `10-neural_network.py`):

- Main file: `11-main.py`
- Add the public method `def cost(self, Y, A)`:
  - Calculates the cost of the model using logistic regression
  - `Y` is a `numpy.ndarray` with shape (1, `m`) that contains the correct labels for the input data
  - `A` is a `numpy.ndarray` with shape (1, `m`) containing the activated output of the neuron for each example
  - To avoid division by zero errors, please use `1.0000001 - A` instead of `1 - A`
  - Returns the cost

```bash
alexa@ubuntu-xenial:$ ./11-main.py
0.7917984405648548
alexa@ubuntu-xenial:$
```

### 12. Evaluate NeuralNetwork

Write a class `NeuralNetwork` that defines a neural network with one hidden layer performing binary classification (based on `11-neural_network.py`):

- Main file: `12-main.py`
- Add the public method `def evaluate(self, X, Y)`:
  - Evaluates the neuron's predictions
  - `X` is a `numpy.ndarray` with shape (nx, `m`) that contains the input data
    - `nx` is the number of input features to the neuron
    - `m` is the number of examples
  - `Y` is a `numpy.ndarray` with shape (1, `m`) that contains the correct labels for the input data
  - Returns the neuron’s prediction and the cost of the network, respectively
    - The prediction should be a `numpy.ndarray` with shape (1, `m`) containing the predicted labels for each example
    - The label values should be 1 if the output of the network is >= 0.5 and 0 otherwise

```bash
alexa@ubuntu-xenial:$ ./12-main.py
[[0 0 0 ... 0 0 0]]
0.7917984405648548
alexa@ubuntu-xenial:$
```

### 13. NeuralNetwork Gradient Descent

Write a class `NeuralNetwork` that defines a neural network with one hidden layer performing binary classification (based on `12-neural_network.py`):

- Main file: `13-main.py`
- Add the public method `def gradient_descent(self, X, Y, A1, A2, alpha=0.05)`:
  - Calculates one pass of gradient descent on the neural network
  - `X` is a `numpy.ndarray` with shape (nx, `m`) that contains the input data
    - `nx` is the number of input features to the neuron
    - `m` is the number of examples
  - `Y` is a `numpy.ndarray` with shape (1, `m`) that contains the correct labels for the input data
  - `A1` is the output of the hidden layer
  - `A2` is the predicted output
  - `alpha` is the learning rate
  - Updates the private attributes `__W1`, `__b1` `__W2`, and `__b2`

```bash
alexa@ubuntu-xenial:$ ./13-main.py
[[ 1.76405235  0.40015721  0.97873798 ...  0.52130375  0.61192719
  -1.34149673]
 [ 0.47689837  0.14844958  0.52904524 ...  0.0960042  -0.0451133
   0.07912172]
 [ 0.85053068 -0.83912419 -1.01177408 ... -0.07223876  0.31112445
  -1.07836109]]
[[ 0.003193  ]
 [-0.01080922]
 [-0.01045412]]
[[ 1.06583858 -1.06149724 -1.79864091]]
[[0.15552509]]
alexa@ubuntu-xenial:$
```

### 14. Train NeuralNetwork

Write a class `NeuralNetwork` that defines a neural network with one hidden layer performing binary classification (based on `13-neural_network.py`):

- Main file: `14-main.py`
- Add the public method `def train(self, X, Y, iterations=5000, alpha=0.05)`:
  - Trains the neural network
    - `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
      - `nx` is the number of input features to the neuron
      - `m` is the number of examples
    - `Y` is a `numpy.ndarray` with shape (1, `m`) that contains the correct labels for the input data
    - `iterations` is the number of iterations to train over
      - if `iterations` is not an integer, raise a `TypeError` with the exception `iterations must be an integer`
      - if `iterations` is not positive, raise a `ValueError` with the exception `iterations must be a positive integer`
    - `alpha` is the learning rate
      - if `alpha` is not a float, raise a `TypeError` with the exception `alpha must be a float`
      - if `alpha` is not positive, raise a `ValueError` with the exception `alpha must be positive`
    - All exceptions should be raised in the order listed above
    - Updates the private attributes `__W1`, `__b1`, `__A1`, `__W2`, `__b2`, and `__A2`
    - You are allowed to use one loop
    - Returns the evaluation of the training data after `iterations` of training have occurred

```bash
alexa@ubuntu-xenial:$ ./14-main.py
Train cost: 0.4680930945144984
Train accuracy: 84.69009080142123%
Dev cost: 0.45985938789496067
Dev accuracy: 86.52482269503547%
alexa@ubuntu-xenial:$
```

### 15. Upgrade Train NeuralNetwork

Write a class `NeuralNetwork` that defines a neural network with one hidden layer performing binary classification (based on `14-neural_network.py`):

- Main file: `15-main.py`
- Update the public method train to `def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100)`:
  - Trains the neural network
  - `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
    - `nx` is the number of input features to the neuron
    - `m` is the number of examples
  - `Y` is a `numpy.ndarray` with shape (1, `m`) that contains the correct labels for the input data
  - `iterations` is the number of iterations to train over
    - if `iterations` is not an integer, raise a `TypeError` with the exception `iterations must be an integer`
    - if `iterations` is not positive, raise a `ValueError` with the exception `iterations must be a positive integer`
  - `alpha` is the learning rate
    - if `alpha` is not a float, raise a `TypeError` with the exception `alpha must be a float`
    - if `alpha` is not positive, raise a `ValueError` with the exception `alpha must be positive`
  - Updates the private attributes `__W1`, `__b1`, `__A1`, `__W2`, `__b2`, and `__A2`
  - `verbose` is a boolean that defines whether or not to print information about the training. If `True`, print `Cost after {iteration} iterations: {cost}` every `step` iterations:
    - Include data from the 0th and last iteration
  - `graph` is a boolean that defines whether or not to graph information about the training once the training has completed. If `True`:
    - Plot the training data every `step` iterations as a blue line
    - Label the x-axis as `iteration`
    - Label the y-axis as `cost`
    - Title the plot `Training Cost`
    - Include data from the 0th and last iteration
  - Only if either `verbose` or `graph` are `True`:
    - if `step` is not an integer, raise a `TypeError` with the exception `step must be an integer`
    - if `step` is not positive or is greater than iterations, raise a `ValueError` with the exception `step must be positive and <= iterations`
  - All exceptions should be raised in the order listed above
  - The 0th iteration should represent the state of the neuron before any training has occurred
  - You are allowed to use one loop
  - You can use `import matplotlib.pyplot as plt`
  - Returns: the evaluation of the training data after `iterations` of training have occurred

```bash
alexa@ubuntu-xenial:$ ./15-main.py
Cost after 0 iterations: 0.7917984405648547
Cost after 100 iterations: 0.4680930945144984

...

Cost after 5000 iterations: 0.024369225667283875

...

Train cost: 0.024369225667283875
Train accuracy: 99.3999210422424%
Dev cost: 0.020330639788072768
Dev accuracy: 99.57446808510639%
```

```bash

```
