#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from perceptron import PerceptronClassifier
from smo import SVMClassifier
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

class Perceptron:

    def __init__(self):
        iris = datasets.load_iris()

        x = iris.data[0:150]
        t = iris.target[0:150]
        for i in range(150):
            if t[i] == 0:
                t[i] = -1
            if t[i] == 2:
                t[i] = 1

        self.classifier = PerceptronClassifier(4, 0.1)
        self.classifier.learn(t, x)

    def predict(self, x):
        y = self.classifier.classify(x)
        if y == -1:
            return 'setosa'
        else:
            return None

class SVM:

    def __init__(self):
        iris = datasets.load_iris()

        x = iris.data[50:150]
        t = iris.target[50:150]
        for i in range(100):
            if t[i] == 2:
                t[i] = -1

        self.classifier = SVMClassifier(4, 0.1)
        self.classifier.learn(t, x)

    def predict(self, x):
        y = self.classifier.classify(x)
        if y == -1:
            return 'virginica'
        else:
            return 'versicolor'

class IRISClassifier:

    def __init__(self):
        self.perceptron = Perceptron()
        self.svm = SVM()

    def predict(self, x):
        predict = self.perceptron.predict(x)
        if predict:
            return predict
        predict = self.svm.predict(x)
        return predict

def main():
    iris = datasets.load_iris()
    classifier = IRISClassifier()

    data = 6 * np.random.random_sample((100,4))

    setosa = []
    versicolor = []
    virginica = []
    for i in range(150):
        predict = classifier.predict(iris.data[i])
        if predict == 'setosa':
            setosa.append(i)
        elif predict == 'versicolor':
            versicolor.append(i)
        elif predict == 'virginica':
            virginica.append(i)

    plt.figure(figsize=(20, 8))

    for i, (x, y) in enumerate([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]):
        plt.subplot(2, 3, i + 1)

        plt.scatter(
            iris.data[setosa, x],
            iris.data[setosa, y],
            marker = '>',
            c = 'r',
            label = 'setosa',
        )
        plt.scatter(
            iris.data[versicolor, x],
            iris.data[versicolor, y],
            marker = 'o',
            c = 'g',
            label = 'versicolor',
        )
        plt.scatter(
            iris.data[virginica, x],
            iris.data[virginica, y],
            marker = 'x',
            c = 'b',
            label = 'virginica',
        )
        plt.xlabel(iris.feature_names[x])
        plt.ylabel(iris.feature_names[y])
        plt.autoscale()
        plt.grid()

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show()

if __name__ == "__main__":
    main()
