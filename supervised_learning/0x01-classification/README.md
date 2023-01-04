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
