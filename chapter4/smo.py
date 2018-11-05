#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random
import sys
import numpy as np

class SVMClassifier(object):
    """ SVM + SMO """

    def __init__(self, dim, C = 0.01):
        self.C = C
        self.eps = 0.01
        self.tol = 0.01
        self.a = None
        self.point = None
        self.target = None
        self.E = None
        self.b = 0.0
        self.N = dim

    def takeStep(self, i1, i2):
        if i1 == i2: return False

        alph1 = self.a[i1]
        alph2 = self.a[i2]
        y1 = self.target[i1]
        y2 = self.target[i2]
        E1 = self.E[i1]
        E2 = self.E[i2]

        if alph1 > 0 and alph1 < self.C:
          E1 = self.E[i1]
        else:
          E1 = self.learned_func(i1) - y1

        if alph2 > 0 and alph2 < self.C:
          E2 = self.E[i2]
        else:
          E2 = self.learned_func(i2) - y2

        s = y1 * y2
        L, H = 0.0, 0.0

        if y1 == y2:
            gamma = alph1 + alph2
            if gamma > self.C:
                L = gamma - self.C
                H = self.C
            else:
                L = 0
                H = gamma
        else:
            gamma = alph1 - alph2
            if gamma > 0:
                L = 0
                H = self.C - gamma
            else:
                L = -gamma
                H = self.C

        if L == H: return False

        k11 = self.kernel(self.point[i1], self.point[i1])
        k12 = self.kernel(self.point[i1], self.point[i2])
        k22 = self.kernel(self.point[i2], self.point[i2])

        eta = 2 * k12 - k11 - k22
        a1, a2 = 0.0, 0.0

        if eta < 0:
            a2 = alph2 + y2 * (E2 - E1) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            c1 = eta / 2
            c2 = y2 * (E1 - E2) - eta * alph2
            Lobj = c1 * L * L + c2 * L
            Hobj = c1 * H * H + c2 * H

            if Lobj > Hobj + self.eps:
                a2 = L
            elif Lobj < Hobj - self.eps:
                a2 = H
            else:
                a2 = alph2

        if abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps):
            return False

        a1 = alph1 - s * (a2 - alph2)
        if a1 < 0:
            a2 += s * a1
            a1 = 0
        elif a1 > self.C:
            t = a1 - self.C
            a2 += s * t
            a1 = self.C

        bnew = 0.0
        if a1 > 0 and a1 < self.C:
            bnew = self.b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12
        else:
            if a2 > 0 and a2 < self.C:
                bnew = self.b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22
            else:
                b1 = self.b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12
                b2 = self.b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22
                bnew = (b1 + b2) / 2
        delta_b = bnew - self.b
        self.b = bnew

        t1 = y1 * (a1 - alph1)
        t2 = y2 * (a2 - alph2)

        for i in range(self.N):
          if self.a[i] > 0 and self.a[i] < self.C:
              self.E[i] += t1 * self.kernel(self.point[i1], self.point[i]) + t2 * self.kernel(self.point[i2], self.point[i]) - delta_b

        self.E[i1] = 0.0
        self.E[i2] = 0.0

        self.a[i1] = a1
        self.a[i2] = a2

        return True


    def examineExample(self, i1):
        y1 = self.target[i1]
        alph1 = self.a[i1]
        E1 = self.E[i1]
        i2 = -1

        if alph1 > 0 and alph1 < self.C:
            E1 = self.E[i1]
        else:
            E1 = self.learned_func(i1) - y1

        r1 = y1 * E1

        if (r1 < -self.tol and alph1 < self.C) or (r1 > self.tol and alph1 > 0):

            max_val = 0
            for i in range(self.N):
                if self.a[i] > 0 and self.a[i] < self.C:
                  if abs(E1 - self.E[i]) > max_val:
                        max_val = abs(E1 - self.E[i])
                        i2 = i

            if i2 >= 0:
                if self.takeStep(i1, i2):
                    return 1

            s = np.random.randint(0, self.N) * self.N
            for i in range(s, s + self.N):
                j = i % self.N
                if self.a[j] > 0 and self.a[j] < self.C:
                    i2 = j
                    if self.takeStep(i1, i2):
                        return 1

            s = np.random.randint(0, self.N) * self.N
            for i in range(s, s + self.N):
                i2 = i % self.N
                if self.takeStep(i1, i2):
                    return 1

        return 0


    def learn(self, y, x):
        self.N = len(x)

        self.a = np.zeros(self.N)
        self.point = np.array(x)
        self.target = np.array(y)
        self.E = -1.0 * np.array(y)

        numChanged = 0
        examineAll = True

        while numChanged > 0 or examineAll:
            numChanged = 0
            if examineAll:
                for i in range(self.N):
                    numChanged += self.examineExample(i)
            else:
                for i in range(self.N):
                    if self.a[i] != 0 and self.a[i] != self.C:
                        numChanged += self.examineExample(i)

            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True

        self.w = np.dot((self.a * self.target).T, self.point)
    
    
    def learned_func(self, k):
        s = 0
        for i in range(self.N):
            if (self.a[i] > 0):
                s += self.a[i] * self.target[i] * self.kernel(self.point[i], self.point[k])
        s -= self.b
        return s

    def kernel(self, x1, x2):
        return np.dot(x1, x2).sum()


    def predict(self, x):
        tmp = 0
        for i in range(self.N):
            tmp += self.a[i] * self.target[i] * self.kernel(x, self.point[i])

        return tmp - self.b


    def classify(self, x):
        pred = self.predict(x)
        if pred > 0:
            return 1
        else:
            return -1

class RbfSVMClassifier(SVMClassifier):

    def __init__(self, dim, C = 1.0, delta = 1.0):
        super(RbfSVMClassifier, self).__init__(dim, C)
        self.delta = delta

    # 動径基底関数カーネル
    def kernel(self, x1, x2):
        return np.exp(- (np.linalg.norm(x1 - x2) ** 2) / (2.0 * (self.delta ** 2)))


class PolySVMClassifier(SVMClassifier):

    def __init__(self, dim, C = 1.0, r = 1.0, d = 2):
        super(PolySVMClassifier, self).__init__(dim, C)
        self.r = r
        self.d = d

    # 多項式カーネル
    def kernel(self, x1, x2):
        return (np.dot(x1, x2) + self.r) ** self.d


def main():
    from matplotlib import pyplot

    # データ
    X = np.random.randn(30, 2)
    T = np.array([1 if x**2 + y ** 2 < 1 else -1 for x, y in X])

    for i in range(30):
      for j in range(2):
          print "%d:%f" % (j, X[i][j]),

    classifier = SVMClassifier(2, 10)

    classifier.learn(T, X)

    XX = np.random.randn(100, 2)
    TT = np.array([1 if x**2 + y ** 2 < 1 else -1 for x, y in XX])

    for i in range(30):
      ret = 0
      if classifier.predict(X[i]) > 0:
        ret = 1
      else:
        ret = -1

      if T[i] * ret > 0:
        print "true"
      else:
        print "false"


    param = sys.argv
    if '-d' in param or '-s' in param:
        seq = np.arange(-3, 3, 0.02)
        pyplot.figure(figsize = (6, 6))
        pyplot.xlim(-3, 3)
        pyplot.ylim(-3, 3)
        pyplot.plot(seq, -(classifier.w[0] * seq - classifier.b) / classifier.w[1], 'k-')
        pyplot.plot(X[T ==  1,0], X[T ==  1,1], 'ro')
        pyplot.plot(X[T == -1,0], X[T == -1,1], 'bo')

        if '-s' in param:
            pyplot.savefig('graph.png')

        if '-d' in param:
            pyplot.show()

if __name__ == "__main__":
    main()
