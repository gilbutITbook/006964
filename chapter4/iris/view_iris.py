#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from sklearn import datasets

def main():
    iris = datasets.load_iris()

    plt.figure(figsize=(20, 8))

    for i, (x, y) in enumerate([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]):
        plt.subplot(2, 3, i + 1)
        for t, marker, c in [(0,'>','r'),(1,'o','g'),(2,'x','b')]:

            plt.scatter(
                iris.data[iris.target == t, x],
                iris.data[iris.target == t, y],
                marker = marker,
                c = c,
                label = iris.target_names[t],
            )
            plt.xlabel(iris.feature_names[x])
            plt.ylabel(iris.feature_names[y])
            plt.autoscale()
            plt.grid()

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show()

if __name__ == '__main__':
    main()
