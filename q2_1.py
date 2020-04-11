#!/bin/python

import numpy as np
import csv
import math as m
import matplotlib.pyplot as plt
import sys

def get_acc(w, X, y):

        acc_predicts = 0

        for i in range(len(X)):

                if y[i] == 1:

                        if (1.0 / (1.0 + np.exp(-np.dot(np.transpose(w), X[i])))) >= 0.5:
                                acc_predicts += 1

                elif y[i] == 0:

                        if (1 - (1.0 / (1.0 + np.exp(-np.dot(np.transpose(w), X[i]))))) >= 0.5:
                                acc_predicts += 1

        return acc_predicts / len(X)

if __name__ == '__main__':

        traincsv = str(sys.argv[1])
        testcsv = str(sys.argv[2])
        trainpath = "./{}".format(traincsv)
        testpath = "./{}".format(testcsv)

        f = open(trainpath, "r")
        train = list(csv.reader(f, delimiter=","))
        train = np.array(train[0:], dtype=np.float)

        p_train = np.delete(train, 256, 1)
        p_train = p_train/255
        p_train = np.insert(p_train, 0, 1, axis=1)

        a_train = np.delete(train, np.s_[0:256], 1)

        f = open(testpath, "r")
        test = list(csv.reader(f, delimiter=","))
        test = np.array(test[0:], dtype=np.float)

        p_test = np.delete(test, 256, 1)
        p_test = p_test/255
        p_test = np.insert(p_test, 0, 1, axis=1)

        a_test = np.delete(test, np.s_[0:256], 1)

        w_log = np.zeros(p_train.shape[1])
        eps = m.exp(-3)
        nu = 0.0001
        k = 0
        lam = [0.001, 0.01, 1, 10, 100, 1000]
        num_batches = 100

        ll_ot = [[], [], []]

        while True:
                grad_des = np.zeros(p_train.shape[1])
                for i in range(p_train.shape[0]):
                        pred_ans = 1.0 / (1.0 + m.exp(-np.dot(np.transpose(w_log), p_train[i])))
                        grad_des += ((pred_ans - a_train[i]) * p_train[i])

                w_log -= (nu*(grad_des + (lam[5]*w_log)))
                k += 1

                ll_ot[0].append(k)
                ll_ot[1].append(get_acc(w_log, p_train, a_train))
                ll_ot[2].append(get_acc(w_log, p_test, a_test))

                if k == num_batches:
                        break

        fig, ax = plt.subplots()

        ax.plot(ll_ot[0], ll_ot[1], label="Training Set")
        ax.plot(ll_ot[0], ll_ot[2], label="Testing Set")
        ax.legend(loc="lower right")
        plt.show()

