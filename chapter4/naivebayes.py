#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import sys
from collections import defaultdict

class Feature:
    """ Feature for NaiveBayes"""

    def __init__(self, name):
        self.name = name
        self.categories = defaultdict(int)

    def __getitem__(self, category):
        return self.categories[category]

    def __setitem__(self, category, value):
        self.categories[category] = value


class NaiveBayesClassifier:
    """ NaiveBayes """

    def __init__(self):
        self.categories = defaultdict(int)
        self.features = {}
        self.training_count = 0
        self.alpha = 1


    def learn(self, category, features):
        self.categories[category] += 1
        self.training_count += 1

        for f in features:
            if f not in self.features:
                self.features[f] = Feature(f)
            self.features[f][category] += 1


    def classifly(self, features):
        result = None
        max_score = 0

        for c in self.categories:
            score = float(self.categories[c] + self.alpha) / (self.training_count + len(self.categories) * self.alpha)

            for f in self.features:
                score *= float(self.features[f][c] + self.alpha) / (self.categories[c] + 2 * self.alpha)

            if max_score < score:
              result, max_score = c, score

        return result


    def get_alpha(self):
        return self.alpha

    def set_alpha(self, value):
        self.alpha = value


def main():
    # データ
    training_data = [["good", [u"좋다", u"매우"]],
                   ["good", [u"좋다", u"매우", u"굉장하다"]],
                   ["good", [u"좋다", u"굉장하다", u"찾을 수 없다"]], 
                   ["good", [u"굉장하다"]],
                   ["bad",  [u"찾을 수 없다", u"사고 싶지 않다"]],
                   ["bad",  [u"좋다"]],
                   ["bad",  [u"사고 싶지 않다", u"최악"]],
                   ["bad",  [u"최악"]]]
    test_data  = [u"좋다", u"매우"]


    classifier = NaiveBayesClassifier()

    # 学習フェーズ
    for c, f in training_data:
      classifier.learn(c, f)

    # 適用フェーズ
    print classifier.classifly(test_data)

if __name__ == "__main__":
    main()
