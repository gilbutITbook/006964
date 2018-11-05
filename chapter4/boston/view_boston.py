#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from sklearn import datasets

def main():
    boston = datasets.load_boston()

    plt.figure(figsize=(30, 8))

    for i in range(13):
        plt.subplot(2, 7, i + 1)
        plt.scatter(
            boston.data[:, i],
            boston.target,
            marker = 'o',
            c = 'r',
        )
        plt.xlabel(boston.feature_names[i])
        plt.ylabel('house price')
        plt.autoscale()
        plt.grid()

    plt.show()

if __name__ == '__main__':
    main()
