#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import sys
import numpy as np

class PerceptronOnlineClassifier(object):
    """ Perceptron """

    def __init__(self, size, alpha=1.0):
        self.w = np.random.uniform(-1.0, 1.0, size + 1)
        self.alpha = alpha

    def learn(self, y, x):
        margin = self.margin(x)

        if y * self.margin(x) < 0:
            eta = self.eta(y, x)
            self.w = self.w + self.alpha * eta * np.append(x, 1)
            return True
        else:
            return False

    def eta(self, y, x):
        return y

    def margin(self, x):
        return np.dot(self.w, np.append(x, 1))

    def classify(self, x):
        if self.margin(x) > 0:
            return 1
        else:
            return -1


def main():
    # データ
    training_data = [[-1, [-1, -1]],
                     [ 1, [-1,  1]],
                     [ 1, [ 1, -1]],
                     [ 1, [ 1,  1]]]

    test_data = [2, 3]

    classifier = PerceptronOnlineClassifier(2, 0.1)

    # 学習フェーズ
    for y, x in training_data:
        classifier.learn(y, x)

    # 適用フェーズ
    print classifier.classify(test_data)

if __name__ == "__main__":
    main()