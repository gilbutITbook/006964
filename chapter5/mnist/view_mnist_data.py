#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    X = mnist.train.images
    y = mnist.train.labels

    for i in range(0, 5):
        plt.axis('off')
        plt.subplot(5, 1, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap=cm.gray_r, interpolation='nearest')
        plt.title(y[i], color='red')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
