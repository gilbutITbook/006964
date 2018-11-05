#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np

def main():
    boston = datasets.load_boston()

    prices = boston.target

    x = np.array([np.concatenate((v, [1])) for v in boston.data])
    y = prices
    slope, total_error, _, _ = np.linalg.lstsq(x, y)

    for i in range(10):
        predict = np.dot(slope, np.concatenate((boston.data[i], [1])))
        print i, 'predict=', predict, 'target=', boston.target[i]

if __name__ == '__main__':
    main()
