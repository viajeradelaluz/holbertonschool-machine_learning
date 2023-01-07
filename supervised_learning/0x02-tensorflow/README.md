# Tensorflow

## Learning Objectives

- What is tensorflow?
  What is a session? graph?
- What are tensors?
- What are variables? constants? placeholders? How do you use them?
- What are operations? How do you use them?
- What are namespaces? How do you use them?
- How to train a neural network in tensorflow
- What is a checkpoint?
- How to save/load a model with tensorflow
- What is the graph collection?
- How to add and get variables from the collection

## Tasks

### 0. Placeholders

Write the function `def create_placeholders(nx, classes)`: that returns two placeholders, `x` and `y`, for the neural network:

- Main file: `0-main.py`
- `nx`: the number of feature columns in our data
- `classes`: the number of classes in our classifier
- Returns: placeholders named `x` and `y`, respectively
  - `x` is the placeholder for the input data to the neural network
  - `y` is the placeholder for the one-hot labels for the input data

### 1. Layers

Write the function `def create_layer(prev, n, activation)`:

- Main file: `1-main.py`
- `prev` is the tensor output of the previous layer
- `n` is the number of nodes in the layer to create
- `activation` is the activation function that the layer should use
- use `tf.keras.initializers.VarianceScaling(mode='fan_avg')` to implement `He et. al` initialization for the layer weights
- each layer should be given the name `layer`
- Returns: the tensor output of the layer

### 2. Forward Propagation

Write the function `def forward_prop(x, layer_sizes=[], activations=[])`: that creates the forward propagation graph for the neural network:

- Main file: `2-main.py`
- `x` is the placeholder for the input data
- `layer_sizes` is a list containing the number of nodes in each layer of the network
- `activations` is a list containing the activation functions for each layer of the network
- Returns: the prediction of the network in tensor form
- For this function, you should import your `create_layer` function with `create_layer = **import**('1-create_layer').create_layer`

### 3. Accuracy

Write the function `def calculate_accuracy(y, y_pred)`: that calculates the accuracy of a prediction:

- Main file: `3-main.py`
- `y` is a placeholder for the labels of the input data
- `y_pred` is a tensor containing the network’s predictions
- Returns: a tensor containing the decimal accuracy of the prediction

  hint: accuracy = correct_predictions / all_predictions

### 4. Loss

Write the function `def calculate_loss(y, y_pred)`: that calculates the softmax cross-entropy loss of a prediction:

- Main file: `4-main.py`
- `y` is a placeholder for the labels of the input data
- `y_pred` is a tensor containing the network’s predictions
- Returns: a tensor containing the loss of the prediction

### 5. Train_Op

Write the function `def create_train_op(loss, alpha)`: that creates the training operation for the network:

- Main file: `5-main.py`
- `loss` is the loss of the network’s prediction
- `alpha` is the learning rate
- Returns: an operation that trains the network using gradient descent
