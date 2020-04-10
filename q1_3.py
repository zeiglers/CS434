#!/bin/python

import numpy as np
import csv
#import math as m
#import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':

        traincsv = str(sys.argv[1])
        testcsv = str(sys.argv[2])
        trainpath = "./{}".format(traincsv)
        testpath = "./{}".format(testcsv)

        #getting w
        f = open(trainpath, "r")
        train = list(csv.reader(f, delimiter=","))
        train = np.array(train[0:], dtype=np.float)


        xtrain = np.delete(train, 13,axis= 1)
        ytrain = np.delete(train, np.s_[0:13], axis=1)


        xtraint = xtrain.transpose()
        w = np.dot(np.dot(np.linalg.inv(np.dot(xtraint, xtrain)), xtraint), ytrain)

        #getting ase of train data

        pre_train = np.dot(xtrain, w)
        sse_train = np.subtract(ytrain, pre_train)
        n_train = sse_train.shape[0]
        sse_train = np.dot(sse_train.transpose(), sse_train)
        ase_train = sse_train/n_train



        #getting ase of test data
        f = open(testpath, "r")
        test = list(csv.reader(f, delimiter=","))
        test = np.array(test[0:], dtype=np.float)


        xtest = np.delete(test, 13,axis= 1)
        ytest = np.delete(test, np.s_[0:13], axis=1)

        pre_test = np.dot(xtest, w)
        sse_test = np.subtract(ytest, pre_test)
        n_test = sse_test.shape[0]
	sse_test = np.dot(sse_test.transpose(), sse_test)
        ase_test = sse_test/n_test

        print("\n")
        print ("The Learned Weight Vector: ({})(X1)".format(round(w[0][0],3))),

        for x in range(1,w.shape[0]):
                m = round(w[x][0],3)
                print("+ ({})(X{})".format(m, x+1)),

        print("\n")
        print("ASE over the training data: {}".format(round(ase_train[0][0],3)))
        print("ASE over the testing data: {}".format(round(ase_test[0][0],3)))


