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
