#!/bin/python

import numpy as np
import csv
import math as m
import matplotlib.pyplot as plt
import sys

def acc_point(log, pred, acc):

        predict = 0
        for i in range(len(pred)):
                if acc[i] == 1:
                        if (1.0 / (1.0 + np.exp(-np.dot(np.transpose(log), pred[i])))) >= 0.5:
                                predict += 1
                elif acc[i] == 0:
                        if (1 - (1.0 / (1.0 + np.exp(-np.dot(np.transpose(log), pred[i]))))) >= 0.5:
                                predict += 1
        return predict / len(pred)

if __name__ == '__main__':

        # Read in file paths for training and testing files
        traincsv = str(sys.argv[1])
        testcsv = str(sys.argv[2])
        trainpath = "./{}".format(traincsv)
        testpath = "./{}".format(testcsv)

        # Read from training data and store usable data
        f = open(trainpath, "r")
        train = list(csv.reader(f, delimiter=","))
        train = np.array(train[0:], dtype=np.float)
        # Remove extra info from data
        p_train = np.delete(train, 256, 1)
        p_train = p_train/255
        p_train = np.insert(p_train, 0, 1, axis=1)
        a_train = np.delete(train, np.s_[0:256], 1)

        # Read from testing data and store usable data
        f = open(testpath, "r")
        test = list(csv.reader(f, delimiter=","))
        test = np.array(test[0:], dtype=np.float)
        # Remove extra info from data
        p_test = np.delete(test, 256, 1)
        p_test = p_test/255
        p_test = np.insert(p_test, 0, 1, axis=1)
        a_test = np.delete(test, np.s_[0:256], 1)

        # Set up to use logistic regression
        log = np.zeros(p_train.shape[1])
        eps = m.exp(-3)
        nu = 0.0001
        k = 0
        l = [0.001, 0.01, 1, 10, 100, 1000]
        batches = 100

        # Initiallize matrix to store plot points
        mx = [[], [], []]

        # While loop for gradient descent
        while True:
                # Begin gradient descent
                descent = np.zeros(p_train.shape[1])
                # Loop through every training item
                for i in range(p_train.shape[0]):
                        a_predic = 1.0 / (1.0 + m.exp(-np.dot(np.transpose(log), p_train[i])))
                        descent += ((a_predic - a_train[i]) * p_train[i])
                log -= (nu*(descent + (l[5]*log)))
                # Add each point to the matrix
                k += 1
                mx[0].append(k)
                mx[1].append(acc_point(log, p_train, a_train))
                mx[2].append(acc_point(log, p_test, a_test))

                if k == batches:
                        break

        # Plot the graphs
        fig, ax = plt.subplots()

        ax.plot(mx[0], mx[1], label="Training")
        ax.plot(mx[0], mx[2], label="Testing")
        ax.legend(loc="lower right")
        plt.show()

