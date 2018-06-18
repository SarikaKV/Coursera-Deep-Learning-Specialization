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
parser.add_argument('--num_iterations', '-num', metavar='num_iterations', default=3001, help='hyperparameter representing the number of iterations to optimize the parameters')
parser.add_argument('--learning_rate', '-alpha', metavar='learning_rate', default=0.009, help='learning rate of the gradient descent update rule')
parser.add_argument('--print_cost', '-print', metavar='print_cost', default=False, help='True to print the loss every 100 steps')
parser.add_argument('--threshold', '-t', metavar='threshold', default=0.45, help='Threshold at which we divide positive and negative examples')


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
    return s

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    """
    w = np.zeros(shape=(dim, 1))
    b = 0
    return w, b

def forward_propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above
    """
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    return A

def compute_cost_and_backpropogate(A,X,Y):
    """
    Implement the cost function and gradients. 
    """
    m = X.shape[1]
    
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    # BACKWARD PROPAGATION (TO FIND GRAD)

    dw = np.dot(X, (A - Y).T)/m
    db = np.sum(A - Y)/m
    
    cost = np.squeeze(cost)
    grads = {"dw": dw, "db": db}
        
    return grads, cost
    
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm.    
    """
    
    costs = []    
    for i in range(num_iterations):    
        # Cost and gradient calculation 
        #print('A: ', forward_propagate(w, b, X, Y))
        grads, cost = compute_cost_and_backpropogate(forward_propagate(w, b, X, Y), X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule 
        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
    
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
                 
    return params, grads, costs    

def predict(w, b,threshold, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    '''    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)    
    for i in range(A.shape[1]):
        # Convert probabilities a[0,i] to actual predictions p[0,i]
        Y_prediction[0, i] = 1 if A[0, i] > threshold else 0    
    return Y_prediction

def model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations, learning_rate,threshold,print_cost):
    # initialize parameters with zeros 
    w, b = initialize_with_zeros(train_set_x.shape[0])

    # Gradient descent 
    parameters, grads, costs = optimize(w, b, train_set_x, train_set_y, num_iterations, learning_rate, print_cost)
    print parameters,costs
        
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
        
    # Predict test/train set examples 
    Y_prediction_test = predict(w, b,threshold, test_set_x)
    Y_prediction_train = predict(w, b,threshold, train_set_x)

    # Print experimental Setup values or hyper parameters
    print("Number of Iterations: {}".format(num_iterations))
    print("Learning Rate: {}".format(learning_rate))
    print("Cut off threshold: {}".format(threshold))

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))

    result = {"costs": costs,
             "Y_prediction_test": Y_prediction_test, 
             "Y_prediction_train" : Y_prediction_train, 
             "w" : w, 
             "b" : b,
             "learning_rate" : learning_rate,
             "num_iterations": num_iterations,
             "threshold": threshold}
    return result

def multiple_learning_rates(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations,threshold, print_cost):
    learning_rates = [0.01,0.005, 0.001,0.0005, 0.0001]
    models = {}
    for i in learning_rates:
        print ("learning rate is: " + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y,num_iterations = num_iterations, learning_rate = i,threshold = threshold,print_cost= print_cost)
        print ('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()

if __name__ == '__main__':
    """Executable code from command line."""
    args = parser.parse_args()

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(train_set_x_orig[25])
    plt.subplot(2,2,2)
    plt.imshow(train_set_x_orig[26])
    plt.subplot(2,2,3)
    plt.imshow(train_set_x_orig[27])
    plt.subplot(2,2,4)
    plt.imshow(train_set_x_orig[28])

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
    # With learning rate as specfied in the arguments
    model(train_set_x, train_set_y, test_set_x, test_set_y, args.num_iterations, args.learning_rate ,args.threshold, args.print_cost)

    # Running on multiple learning rates to plot the iterations Vs cost.
    multiple_learning_rates(train_set_x, train_set_y, test_set_x, test_set_y, args.num_iterations, args.threshold,args.print_cost)