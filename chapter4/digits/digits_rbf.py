#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import multiprocessing as mp
from smo import RbfSVMClassifier
from sklearn import datasets, model_selection
import numpy as np

def learn(arg):
    n = arg[0]
    train = arg[1]

    digits = datasets.load_digits()

    x = digits.data[0:]
    t = digits.target[0:]
    for i in range(1797):
        if t[i] == n:
            t[i] = 1
        else:
            t[i] = -1

    classifier = RbfSVMClassifier(64, 0.1, 1.0)
    classifier.learn(t[train], x[train])

    print "learning finished! ", n

    return classifier

def main():

    digits = datasets.load_digits()

    x = digits.data[0:]
    t = digits.target[0:]

    skf = model_selection.StratifiedKFold(n_splits=10)

    pool = mp.Pool(3)
    for train, test in skf.split(x, t):
        classifiers = pool.map(learn, [(0, train), (1, train), (2, train), (3, train), (4, train), (5, train), (6, train), (7, train), (8, train), (9, train)])

        ret = np.zeros([10,10])
        for test_x, test_t in zip(x[test], t[test]):
            for i in range(10):
                test_y = classifiers[i].classify(test_x)
                if test_y == 1:
                    ret[test_t][i] += 1
        print ret
        break

if __name__ == "__main__":
    main()
