#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import sys
import numpy as np
from matplotlib import pyplot

class NeuralNetwork:
    """ 3 layer neural network """

    def __init__(self, input_size, hidden_size, output_size, alpha=0.1):
        # 中間層と出力層の重みの初期化
        self.hw = np.random.random_sample((hidden_size, input_size + 1))
        self.ow = np.random.random_sample((output_size, hidden_size + 1))
        # 学習率をセット
        self.alpha = alpha

    def learn(self, data, epoch):
        # 重みを学習する
        # ループ回数は、 epoch で設定する
        for epo in range(epoch):
            for t, x in data:
                self.update_weight(t, x)

    def predict(self, x):
        # ある入力ベクトル x の出力結果を得る
        z, y = self.forward(x)
        return y

    def f(self, vec):
        # 活性化関数 f を定義
        # この実装は、ロジスティックシグモイド関数
        return np.vectorize(lambda x: 1.0 / (1.0 + math.exp(-x)))(vec)

    def f_dash(self, vec):
        # 活性化関数 f の微分を定義
        # この実装は、ロジスティックシグモイド関数
        return np.vectorize(lambda x: (1.0 - x) * x)(vec)

    def forward(self, x):
        # ネットワークを順方向に伝搬して結果を得る

        hidden_x = np.r_[np.array([1.0]), x]
        hidden_u = self.hw.dot(hidden_x)
        hidden_z = self.f(hidden_u)

        output_x = np.r_[np.array([1.0]), hidden_z]
        output_u = self.ow.dot(output_x)
        output_z = self.f(output_u)

        return (hidden_z, output_z)

    def update_weight(self, t, x):
        # 重みを更新する

        hidden_z, output_z = self.forward(x)

        # 修正値を逆伝搬させて、重みを修正する
        output_x = np.r_[np.array([1.0]), hidden_z]
        output_delta = (output_z - t) * self.f_dash(output_z)
        self.ow -= self.alpha * output_delta.reshape((-1, 1)) * output_x

        hidden_x = np.r_[np.array([1.0]), x]
        hidden_delta = (self.ow[:, 1:].T.dot(output_delta)) * self.f_dash(hidden_z)
        self.hw -= self.alpha * hidden_delta.reshape((-1, 1)) * hidden_x


def main():
    data = np.array([[[1, 0], [0, 0]],
                     [[0, 1], [0, 1]],
                     [[0, 1], [1, 0]],
                     [[1, 0], [1, 1]]])

    input_size = 2
    hidden_size = 2
    output_size = 2
    alpha = 0.1
    epoch = 10000

    nn = NeuralNetwork(input_size, hidden_size, output_size, alpha)
    nn.learn(data, epoch)

    X = np.array([[0,0], [0,1], [1,0], [1,1]])

    for i in range(4):
        x = X[i, :]
        y = nn.predict(x)

        print x
        print y
        print ""

if __name__ == "__main__":
    main()