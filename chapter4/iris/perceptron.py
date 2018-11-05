#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import sys
import numpy as np

class PerceptronClassifier:
    """ Perceptron """

    def __init__(self, size, alpha):
        self.alpha = alpha
        self.weight = np.random.uniform(-1.0, 1.0, size + 1)


    def learn(self, t, x):
        updated = True

        while updated:
            updated = False

            for category, features in zip(t, x):
                predict = self.classify(features)

                if predict != category:
                    self.weight = self.weight + self.alpha * category * np.append(features, 1)
                    updated = True


    def classify(self, features):
        score = np.dot(self.weight, np.append(features, 1))

        if score >= 0:
            return 1
        else:
            return -1


def main():
    # データ
    x = [[-1, -1],
         [-1,  1],
         [ 1, -1],
         [ 1,  1]]
    t = [-1,1,1,1]

    test_data = [2, 3]

    classifier = PerceptronClassifier(2, 0.1)

    # 学習フェーズ
    classifier.learn(t, x)

    # 適用フェーズ
    print classifier.classify(test_data)

if __name__ == "__main__":
    main()
