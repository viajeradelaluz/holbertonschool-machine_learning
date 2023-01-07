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

## 0. Placeholders

Write the function `def create_placeholders(nx, classes)`: that returns two placeholders, `x` and `y`, for the neural network:

- Main file: `0-main.py`
- `nx`: the number of feature columns in our data
- `classes`: the number of classes in our classifier
- Returns: placeholders named `x` and `y`, respectively
  - `x` is the placeholder for the input data to the neural network
  - `y` is the placeholder for the one-hot labels for the input data

## Layers

Write the function `def create_layer(prev, n, activation)`:

- Main file: `1-main.py`
- `prev` is the tensor output of the previous layer
- `n` is the number of nodes in the layer to create
- `activation` is the activation function that the layer should use
- use `tf.keras.initializers.VarianceScaling(mode='fan_avg')` to implement `He et. al` initialization for the layer weights
- each layer should be given the name `layer`
- Returns: the tensor output of the layer
