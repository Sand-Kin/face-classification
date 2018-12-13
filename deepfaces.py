from pylab import *
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import random
import torch
from torch.autograd import Variable

import torchvision.models as models
import torchvision

import torch.nn as nn

#this is a convolutional neural network trained using gradient descent for face classification written about a year ago.
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
        z = 404
    if set == 'Validation':
        x = 10
        z =59
    if set == 'Test':
        x = 20
        z = 120
    result = 0
    X = np.empty((z, 3, 227, 227))
    x_count = 0
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

            file_name = name_file(set + '227', act, actor, str(i))
            img = misc.imread(file_name)

            new_X = ((np.divide(np.array(img), 255.0)).T)
            if new_X.shape != (3, 227, 227):
                continue
            X[x_count] = ((np.divide(np.array(img),255.0)).T)
            x_count += 1
            if i == 0 and actor == 'Alec Baldwin':
                Y = new_Y
            else:
                Y = np.hstack((Y, new_Y))

    return X, Y


class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)

        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias

        classifier_weight_i = [1, 4, 6]
        for i in classifier_weight_i:
            self.classifier[i].weight = an_builtin.classifier[i].weight
            self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.load_weights()

    def get_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        #x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def accuracy(test_x, test_y, model):
    dtype_float = torch.FloatTensor

    #x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)

    y_pred = model(test_x).data.numpy()
    return np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1))

def train_neural_network(iterations, dim_h, batch_size, activation_fn, learning_rate):
    random.seed(58)
    torch.manual_seed(58)

    MyAlex = MyAlexNet()
    MyAlex.eval()

    train_x, train_y = MULTICLASS_create_X_Y_arrays('Training')
    valid_x, valid_y = MULTICLASS_create_X_Y_arrays('Validation')
    test_x, test_y = MULTICLASS_create_X_Y_arrays('Test')

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    dim_x = 256*6*6
    dim_out = 6

    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        activation_fn,
        torch.nn.Linear(dim_h, dim_out),
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    '''x = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
    x = MyAlex.get_features(x).detach()
    y_classes = Variable(torch.from_numpy(np.argmax(train_y, 0)), requires_grad=False).type(dtype_long)'''

    test_x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
    test_x = MyAlex.get_features(test_x).detach()
    #test_y = Variable(torch.from_numpy(np.argmax(test_y, 0)), requires_grad=False).type(dtype_long)

    x_plt = []
    y_plt = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(iterations):
        #making mini batches
        random.seed(t)
        rand_indx = np.random.permutation(404)
        mini_x = []
        mini_y = []
        for i in range(batch_size):

            mini_x.append(train_x[rand_indx[i]])
            mini_y.append(train_y[:,rand_indx[i]])

        mini_train_x = np.asarray(mini_x)
        mini_train_y = np.asarray(mini_y).T

        x = Variable(torch.from_numpy(mini_train_x), requires_grad=False).type(dtype_float)
        x = MyAlex.get_features(x).detach()
        y_classes = Variable(torch.from_numpy(np.argmax(mini_train_y, 0)), requires_grad=False).type(dtype_long)

        y_pred = model(x)
        loss = loss_fn(y_pred, y_classes)
        model.zero_grad()  # Zero out the previous gradient computation

        loss.backward()  # Compute the gradient
        optimizer.step()  # Use the gradient information to
        # make a step
        #x_plt.append(t+1)
        #y_plt.append(accuracy(test_x,test_y, model)*100)
        print accuracy(test_x, test_y.T, model)



    return test_x, test_y.T, model#, x_plt, y_plt

if __name__ == '__main__':
    # testing variables:

    iterations = 500
    dim_h = 50
    batch_size = 32  # max is 405
    activation_fn = torch.nn.ReLU()  # other options: ELU, ReLU, Tanh
    learning_rate = 1e-5
    #########################3
    #NEED TO REMOVE GET FEATURES NOT MY CODE#


    test_x, test_y, model = train_neural_network(iterations, dim_h, batch_size, activation_fn,
                                                                        learning_rate)
    '''plot(x_plt, y_plt, label='Training')
    #plot(x_plt, y_plt_v, label='Validation')
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    xlabel('Iterations')
    ylabel('Accuracy (%)')
    plt.show()'''
    percent_right = accuracy(test_x, test_y, model)
    print percent_right

    '''dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    model = MyAlexNet()
    model.eval()

    

    x = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
    x = model.features(x).detach()'''
