#!/bin/python

import numpy as np
import csv
import sys
import matplotlib.pyplot as plt


if __name__ == '__main__':

	traincsv = str(sys.argv[1])
	testcsv = str(sys.argv[2])
        trainpath = "./{}".format(traincsv)
        testpath = "./{}".format(testcsv)

       
        f = open(trainpath, "r")
        train = list(csv.reader(f, delimiter=","))
        train = np.array(train[0:], dtype=np.float)


        xtrain = np.delete(train, 13,axis= 1)
        ytrain = np.delete(train, np.s_[0:13], axis=1)


        xtrain = np.insert(xtrain, 0, 1, axis = 1)


	ntrain = xtrain.shape[0]
	
	rtrain = np.random.normal(size=(ntrain, 20))



	f = open(testpath, "r")
        test = list(csv.reader(f, delimiter=","))
        test = np.array(test[0:], dtype=np.float)


        xtest = np.delete(test, 13,axis= 1)
        ytest = np.delete(test, np.s_[0:13], axis=1)

        xtest = np.insert(xtest, 0, 1, axis = 1)
 
	ntest = xtest.shape[0]
	rtest = np.random.normal(size=(ntest,20))
	
	x=[2,4,6,8,10,12,14,16,18,20]
	ytrplot=[]
	yteplot=[]


	
	for x in range(0, 20, 2):
		xtrain = np.append(xtrain, rtrain[:,[x,x+1]], axis = 1)
		xtest = np.append(xtest, rtest[:,[x,x+1]], axis = 1)
		
		
		xtraint = xtrain.transpose()
		w = np.dot(np.dot(np.linalg.inv(np.dot(xtraint, xtrain)), xtraint), ytrain)

		pre_train = np.dot(xtrain, w)
		sse_train = np.subtract(ytrain, pre_train)
		sse_train = np.dot(sse_train.transpose(), sse_train)
		ase_train = sse_train/ntrain

		pre_test = np.dot(xtest, w)
        	sse_test = np.subtract(ytest, pre_test)
        	sse_test = np.dot(sse_test.transpose(), sse_test)
        	ase_test = sse_test/ntest

		ytrplot.append(float(ase_train))
		yteplot.append(float(ase_test))
	
	print(ytrplot)
	print(yteplot)

	x=[2,4,6,8,10,12,14,16,18,20]

	plt.plot(x, ytrplot, label="Training Set")
	plt.plot(x, yteplot, label="Testing Set")		

	plt.xlabel('d')
	plt.ylabel('ASE')
	plt.legend(loc="lower right")
	plt.show()
