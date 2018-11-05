#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random
from pa import PerceptronOnlineClassifier
from sklearn import datasets, model_selection

def main():
    iris = datasets.load_iris()

    x = iris.data[50:150]
    t = iris.target[50:150]
    for i in range(100):
        if t[i] == 2:
            t[i] = -1

    skf = model_selection.StratifiedKFold(n_splits=10)

    avg_accuracy = 0
    for train, test in skf.split(x, t):
        classifier = PerceptronOnlineClassifier(4, 0.1)
        random.shuffle(train)
        for train_x, train_t in zip(x[train], t[train]):
            classifier.learn(train_t, train_x)

        accuracy = 0.0
        for test_x, test_t in zip(x[test], t[test]):
            test_y = classifier.classify(test_x)
            if test_y == test_t:
                accuracy += 1
        accuracy /= len(test)
        avg_accuracy += accuracy
    avg_accuracy /= 10

    print 'average accuracy:', avg_accuracy * 100, '%'

if __name__ == "__main__":
    main()
