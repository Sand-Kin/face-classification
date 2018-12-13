from __future__ import division
from pylab import *
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import random
import torch
from torch.autograd import Variable

#this is a linear regression model trained using gradient descent for face classification written about a year ago.
#It operates using the images downloaded and processed by data_processing.py
#TO DO:
#-clean up
#-comment clearly
#-elaborate on test cases
#-maybe implement a new algorithm for validation..

def name_file(processed, act, actor, line):
    #names the downloaded image file, specifying its location to be saved
    first = '/' + processed + '/'

    second = act

    third = actor + '/'
    fourth = line + '.jpg'

    return os.getcwd()+first+second+third+fourth

def MULTICLASS_create_X_Y_arrays(set):

    actors = ['Alec Baldwin', 'Steve Carell', 'Bill Hader', 'Angie Harmon', 'Lorraine Bracco','Peri Gilpin']

    if set == 'Training':
        x = 70
    if set == 'Validation':
        x = 10
    if set == 'Test':
        x = 20
    result = 0
    for actor in actors:

        if (actor == 'Peri Gilpin') and (set == 'Training'):
            x = 55

        if (actor == 'Alec Baldwin') or (actor == 'Steve Carell') or (actor == 'Bill Hader'):
            act = 'Actors/'
        else:
            act = 'Actresses/'

        new_Y = np.zeros((6, 1))
        new_Y[result] = 1

        result += 1
        for i in range(x):

            file_name = name_file(set, act, actor, str(i))
            img = misc.imread(file_name)
            img = img.flatten()
            new_X = np.array([])
            #new_X = np.append(new_X, 1) #adding bias
            new_X = np.append(new_X, np.array(img) / 255.0)
            new_X = np.vstack(new_X)

            if i == 0 and actor == 'Alec Baldwin':
                X = new_X
                Y = new_Y
            else:
                X = np.hstack((X,new_X))
                Y = np.hstack((Y,new_Y))

    return X.T, Y.T

def accuracy(test_x,test_y, model):
    dtype_float = torch.FloatTensor

    x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
    y_pred = model(x).data.numpy()

    return np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1))

def train_neural_network(iterations, dim_h, batch_size, activation_fn, learning_rate):
    random.seed(58)
    torch.manual_seed(58)
    train_x, train_y = MULTICLASS_create_X_Y_arrays('Training')
    valid_x, valid_y = MULTICLASS_create_X_Y_arrays('Validation')
    test_x, test_y = MULTICLASS_create_X_Y_arrays('Test')

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    dim_x = 32*32
    dim_out = 6

    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        activation_fn,
        torch.nn.Linear(dim_h, dim_out),
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    x_plt = []
    y_plt = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(iterations):
        #making mini batches
        random.seed(t)
        rand_indx = np.random.permutation(405)
        mini_x = []
        mini_y = []
        for i in range(batch_size):

            mini_x.append(train_x[rand_indx[i]])
            mini_y.append(train_y[rand_indx[i]])

        mini_train_x = np.asarray(mini_x)
        mini_train_y = np.asarray(mini_y)

        x = Variable(torch.from_numpy(mini_train_x), requires_grad=False).type(dtype_float)
        y_classes = Variable(torch.from_numpy(np.argmax(mini_train_y, 1)), requires_grad=False).type(dtype_long)

        y_pred = model(x)
        loss = loss_fn(y_pred, y_classes)

        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()  # Compute the gradient
        optimizer.step()  # Use the gradient information to
        # make a step
        x_plt.append(t+1)
        y_plt.append(accuracy(test_x,test_y, model)*100)

    y_plt_v = []
    for t in range(iterations):
        #making mini batches
        random.seed(t)
        rand_indx = np.random.permutation(60)
        mini_x = []
        mini_y = []
        for i in range(10):

            mini_x.append(valid_x[rand_indx[i]])
            mini_y.append(valid_y[rand_indx[i]])

        mini_train_x = np.asarray(mini_x)
        mini_train_y = np.asarray(mini_y)

        x = Variable(torch.from_numpy(mini_train_x), requires_grad=False).type(dtype_float)
        y_classes = Variable(torch.from_numpy(np.argmax(mini_train_y, 1)), requires_grad=False).type(dtype_long)

        y_pred = model(x)
        loss = loss_fn(y_pred, y_classes)

        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()  # Compute the gradient
        optimizer.step()  # Use the gradient information to
        # make a step
        y_plt_v.append(accuracy(test_x, test_y, model) * 100)

    return train_x, train_y, valid_x, valid_y, test_x,test_y, model, x_plt, y_plt, y_plt_v


if __name__ == '__main__':

    #testing variables:
    iterations= 5000
    dim_h = 25
    batch_size = 80 #max is 405
    activation_fn = torch.nn.ELU() #other options: ELU, ReLU, Tanh
    learning_rate = 1e-3

    train_x, train_y, valid_x, valid_y, test_x, test_y, model, x_plt, y_plt, y_plt_v = train_neural_network(iterations, dim_h, batch_size, activation_fn, learning_rate)
    plot(x_plt,y_plt,label='Training')
    plot(x_plt, y_plt_v, label='Validation')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    xlabel('Iterations')
    ylabel('Accuracy (%)')
    plt.show()

    percent_right = accuracy(test_x,test_y, model)
    print 'Test Set Accuracy:'
    print percent_right
    percent_right = accuracy(valid_x, valid_y, model)
    print 'Validation set accuracy:'
    print percent_right
    percent_right = accuracy(train_x, train_y, model)
    print 'Training set accuracy:'
    print percent_right
