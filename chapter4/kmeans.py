#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class kMeans:
    """ k-means clustering """

    def __init__(self, k, dim):
        self.k = k
        self.dim = dim

    def clustering(self, data):
        self.centers = data[np.random.randint(0, len(data), self.k)]
        labels  = np.random.choice([0, 1], self.k)
        old_labels = [l for l in labels]

        while True:
          labels = np.array([self.nearest(d, self.centers) for d in data])

          if np.all(labels == old_labels):
              break
          else:
              old_labels = np.copy(labels)
              self.update_centers(labels, data)

        return labels

    def update_centers(self, labels, data):
        for i in range(self.k):
            self.centers[i] = np.mean(data[np.where(labels == i)], axis=0)

    def nearest(self, x, centers):
        sims = [self.sim(x, c) for c in centers]
        return sims.index(max(sims))

    def sim(self, x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    def centers(self):
        return self.centers
