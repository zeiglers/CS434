import numpy as np
import csv
#import math as m
#import matplotlib.pyplot as plt
import sys

def main():
	f = open("./housing_train.csv", "r")
	train = list(csv.reader(f, delimiter=","))
	train = np.array(train[0:], dtype=np.float)


	xtrain = np.delete(train, 13,axis= 1)
	ytrain = np.delete(train, np.s_[0:13], axis=1)


	xtrain = np.insert(xtrain, 0, 1, axis = 1) 
	
	xtraint = xtrain.transpose()
	w = np.dot(np.dot(np.linalg.inv(np.dot(xtraint, xtrain)), xtraint), ytrain)

	print(w)

"""
	f = open("./housing_test.csv", "r")
        test = list(csv.reader(f, delimiter=","))
        test = np.array(test[0:], dtype=np.float)


        xtest = np.delete(test, 13,axis= 1)
        ytest = np.delete(test, np.s_[0:13], axis=1)

	xtest = np.insert(xtest, 0, 1, axis = 1)

"""

	






if __name__ == '__main__':
	main()
