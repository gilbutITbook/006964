#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from sklearn import datasets

def main():
    digits = datasets.load_digits()

    plt.figure(figsize=(20, 8))

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.figure(1, figsize=(3, 3))
        plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(digits.target[i])
    plt.show()

if __name__ == '__main__':
    main()
