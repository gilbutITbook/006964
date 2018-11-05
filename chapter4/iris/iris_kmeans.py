#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from kmeans import kMeans
from sklearn import datasets
import numpy as np

def main():
    iris = datasets.load_iris()

    x = iris.data
    t = iris.target

    k = int(sys.argv[1])
    kmeans = kMeans(k, 4)
    results = kmeans.clustering(iris.data)

    for i in range(3):
        print iris.target_names[i], results[iris.target == i]

if __name__ == "__main__":
    main()
