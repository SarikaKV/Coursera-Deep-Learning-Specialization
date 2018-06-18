# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 21:47:51 2018
Last modified on Mon Mar 12 18:15:52 2018

@authors: Tavanaei, Venkata Sarika Kondra
"""
import argparse, math
import numpy as np
import matplotlib.pyplot as plt
import h5py

parser = argparse.ArgumentParser(description='Binary Classification using Perceptron')
parser.add_argument('--n_h', '-n_h', metavar='n_h', default=805, help='Number of hidden layers')
parser.add_argument('--num_iterations', '-num', metavar='num_iterations', default=3501, help='hyperparameter representing the number of iterations to optimize the parameters')
parser.add_argument('--learning_rate', '-alpha', metavar='learning_rate', default=0.0001, help='learning rate of the gradient descent update rule')
parser.add_argument('--print_cost', '-print', metavar='print_cost', default=True, help='True to print the loss every 100 steps')
parser.add_argument('--threshold', '-t', metavar='threshold', default=0.5, help='Threshold at which we divide positive and negative examples')


def load_dataset(): # this returns 5 matrices/vectors
	# Training dataset
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
	# Training images (209*64*64*3)
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
	# Training labels (209*1)
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
	
	# Test dataset
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
	# Test data (50*64*64*3)
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
	# Test labels (50*1)
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
	
	#List of class names for test data "cat" "no-cat" not be used for computations
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def sigmoid(z):
    """Compute the sigmoid of z"""
    s = 1 / (1 + np.exp(-z))  
    cache = z
    
    return s, cache  

def relu(Z):
    """Implement the RELU function."""
    
    A = np.maximum(0,Z)    
    cache = Z 
    return A, cache

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """    
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)    
    return dZ

def initialize_parameters(n_x, n_h, n_y):
    """
    This function creates a vector of small random numbers of shape (n_h, n_x) for w1 and w2 and initializes b to 0.
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    #b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    #b2 = np.zeros(shape=(n_y, 1))
    
    #parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    parameters = {"W1": W1,"W2": W2}
                     
    return parameters

def linear_forward(A, W):
    """
    Implements the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W"  ; stored for computing the backward pass efficiently
    """
    #Z = np.dot(W, A) + b
    Z = np.dot(W, A)
    #cache = (A, W, b)
    cache = (A, W)
    return Z, cache

def linear_activation_forward(A_prev, W,  activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W". Outputs: "A, activation_cache".
        #Z, linear_cache = linear_forward(A_prev, W, b)
        Z, linear_cache = linear_forward(A_prev, W)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        #Z, linear_cache = linear_forward(A_prev, W, b)
        Z, linear_cache = linear_forward(A_prev, W)
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    cost = np.squeeze(cost)     
    return cost
    
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    """
    #A_prev, W, b = cache
    A_prev, W = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, cache[0].T) / m
    #db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(cache[1].T, dZ) 
    #print 'db:',db   
    #return dA_prev, dW, db
    return dA_prev, dW

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        #print 'at relu'
        dZ = relu_backward(dA, activation_cache)
        #print 'dZ:',dZ.shape
        
    elif activation == "sigmoid":
        #print 'at sigmoid'
        dZ = sigmoid_backward(dA, activation_cache)
        #print 'dZ:',dZ.shape
    
    #dA_prev, dW, db = linear_backward(dZ, linear_cache)
    #return dA_prev, dW, db
    dA_prev, dW = linear_backward(dZ, linear_cache)
    return dA_prev, dW


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        #print "befor: ",parameters["W" + str(l + 1)]
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        #parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]  
        #print "after: ",parameters["W" + str(l + 1)]      
    return parameters

def predict(X, y,threshold, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    #print 'At predict'
    #print parameters
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))

    W1 = parameters["W1"]
    #b1 = parameters["b1"]
    W2 = parameters["W2"]
    #b2 = parameters["b2"]
    
    # Forward propagation
    #A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
    #probas, caches = linear_activation_forward(A1, W2, b2, 'sigmoid')

    A1, cache1 = linear_activation_forward(X, W1,  'relu')
    probas, caches = linear_activation_forward(A1, W2,  'sigmoid')
    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > threshold:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    return p


def two_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims

    # Initialize parameters dictionary
    parameters = initialize_parameters(n_x, n_h, n_y)
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    #b1 = parameters["b1"]
    W2 = parameters["W2"]
    #b2 = parameters["b2"]
    #print W1.shape,b1.shape,W2.shape,b2.shape
    #print 'parameters before: '
    #print parameters
    # Loop (gradient descent)

    for i in range(0, num_iterations):
        print 'iter: ',i
        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
        #A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        #A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')
        A1, cache1 = linear_activation_forward(X, W1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, 'sigmoid')
        #print 'A1: ',A1.shape,'A2: ', A2.shape
        # Compute cost
        cost = compute_cost(A2, Y)
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        #print 'dA2: ',dA2.shape
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2; also dA0 (not used), dW1, db1".
        #dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA1, dW2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        #print 'dA1: ',dA1.shape,'dW2: ',dW2.shape, 'db2: ',db2.shape
        #dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')
        dA0, dW1 = linear_activation_backward(dA1, cache1, 'relu')
        #print 'db1: ',db1.shape, 'db2: ',db2.shape
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        #grads['db1'] = db1
        grads['dW2'] = dW2
        #grads['db2'] = db2
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        #b1 = parameters["b1"]
        W2 = parameters["W2"]
        #b2 = parameters["b2"]
        #print 'parameters afre: '
        #print parameters        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
       
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


if __name__ == '__main__':
    """Executable code from command line."""
    args = parser.parse_args()

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    plt.figure()
    # plt.subplot(2,2,1)
    # plt.imshow(train_set_x_orig[25])
    # plt.subplot(2,2,2)
    # plt.imshow(train_set_x_orig[26])
    # plt.subplot(2,2,3)
    # plt.imshow(train_set_x_orig[27])
    # plt.subplot(2,2,4)
    # plt.imshow(train_set_x_orig[28])

    m_train = train_set_y.shape[1]
    m_test = test_set_y.shape[1]
    num_px = train_set_x_orig.shape[1]
    #Convert to vector of size (num_px*num_px*3,1)
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    #Standardize
    train_set_x = train_set_x_flatten / 255.0
    test_set_x = test_set_x_flatten / 255.0

    print("Train Set: ")
    print(train_set_x.shape)

    print("Test Set: ")
    print(test_set_x.shape)

    print("Train Label: ")
    print(train_set_y.shape)

    print("Test Label: ")
    print(test_set_y.shape)


    # Start your code here
    n_x = 12288     # num_px * num_px * 3
    n_h = args.n_h
    n_y = 1
    layers_dims = (n_x, n_h, n_y)
    # With learning rate as specfied in the arguments
    parameters = two_layer_model(train_set_x, train_set_y, layers_dims, args.learning_rate, args.num_iterations, args.print_cost)
    predictions_train = predict(train_set_x, train_set_y, args.threshold,parameters)
    predictions_test = predict(test_set_x, test_set_y, args.threshold,parameters)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(predictions_train - train_set_y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(predictions_test - test_set_y)) * 100))   
    # Running on multiple learning rates to plot the iterations Vs cost.
    #multiple_learning_rates(train_set_x, train_set_y, test_set_x, test_set_y, args.num_iterations, args.threshold,args.print_cost)