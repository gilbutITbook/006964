#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np

def main():
    boston = datasets.load_boston()

    rm = boston.data[:, 5]
    prices = boston.target

    x = np.array([[v, 1] for v in rm])
    y = prices
    (slope, bias), total_error, _, _ = np.linalg.lstsq(x, y)


    plt.figure(figsize=(30, 8))
    plt.plot(x[:, 0], slope * x[:, 0] + bias)
    plt.scatter(
        rm,
        prices,
        marker = 'o',
        c = 'r',
    )
    plt.xlabel(boston.feature_names[5])
    plt.ylabel('house price')
    plt.autoscale()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
