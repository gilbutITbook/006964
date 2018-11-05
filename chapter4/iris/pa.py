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
        eta = self.eta(y, x)

        if eta == 0:
            return False
        else:
            self.w = self.w + eta * np.append(x, 1)
            return True

    def eta(self, y, x):
        if y * self.margin(x) > 0:
            return 0
        else:
            return self.alpha * y

    def margin(self, x):
        return np.dot(self.w, np.append(x, 1))

    def classify(self, x):
        if self.margin(x) > 0:
            return 1
        else:
            return -1


class PassiveAggressive(PerceptronOnlineClassifier):
    """ Passive Aggressive """

    def __init__(self, size, alpha=1.0):
        super(PassiveAggressive, self).__init__(size, alpha)

    def eta(self, y, x):
        return max(0, (1 - y * self.margin(x)) / np.dot(np.append(x, 1), np.append(x, 1)))


class PassiveAggressive1(PerceptronOnlineClassifier):
    """ Passive Aggressive """

    def __init__(self, size, alpha=1.0, c=1.0):
        super(PassiveAggressive1, self).__init__(size, alpha)
        self.c = c

    def eta(self, y, x):
        return min(self.c, (1 - y * self.margin(x)) / np.dot(np.append(x, 1), np.append(x, 1)))


class PassiveAggressive2(PerceptronOnlineClassifier):
    """ Passive Aggressive """

    def __init__(self, size, alpha=1.0, c=1.0):
        super(PassiveAggressive2, self).__init__(size, alpha)
        self.c = c

    def eta(self, y, x):
        return (1 - y * self.margin(x)) / (np.dot(np.append(x, 1), np.append(x, 1)) + (1.0 / (2.0 * self.c)))

def main():
    # データ
    training_data = [[-1, [-1, -1]],
                     [ 1, [-1,  1]],
                     [ 1, [ 1, -1]],
                     [ 1, [ 1,  1]]]

    test_data = [2, 3]

    classifier = PassiveAggressive(2, 0.1)

    # 学習フェーズ
    for y, x in training_data:
        classifier.learn(y, x)

    # 適用フェーズ
    print classifier.classify(test_data)

if __name__ == "__main__":
    main()
