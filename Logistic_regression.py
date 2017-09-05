#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 08:09:22 2017

@author: yohei
"""

import numpy as np
from mnist import MNIST


class Logistic_regression(object):
    """
    ロジスティック回帰のクラス
    学習率:eta
    勾配方法:gradient_method
    """
    def __init__(self, eta=0.01, gradient_method='GD', seed=None):
        self.eta = eta
        self.gradient_method = gradient_method
        self.seed = seed

    def fit(self, X, y):
        self.k_class = len(np.unique(y))
        seed = np.random.RandomState(self.seed)
        self.intercept_ = (seed.rand(k_class) - 0.5) / 50
        self.coef_ = (seed.rand(k_class, X.shape[1]) - 0.5) / 50
        T = np.zeros((X_train.shape[0], k_class))
        for ik in range(k_class):
            T[y == ik, ik] = 1

        if self.gradient_method == 'GD':
            self.__GD(X, T)

        elif self.gradient_method == 'SGD':
            self.__SGD(X, T)

        elif self.gradient_method == 'MSGD':
            self.__MSGD(X, T)

        else:
            raise NameError("gradient_method は ['GD', 'SGD', 'MSGD']のうちどれかを指定")

    def score(self, X, y):
        f = self.intercept_ + np.dot(X, self.coef_.T)
        z = self.__softmax(f)
        argmax = np.argmax(z, axis=1)

        discriminative_rate = len(y[argmax == y])
        """
        print('誤識別:{0}番目の {1} を {2} と識別'.format(i, Tlabel[], argmax))
        print('z:', z)
        """
        print('Tに対する識別率: {:.2%}'.format(discriminative_rate / X.shape[0]))

    def __softmax(self, f):
        # softmax関数を返す
        c = np.max(f, axis=1)
        c = c[:, np.newaxis]
        exp_f = np.exp(f - c)
        sum_exp_f = np.sum(exp_f, axis=1)
        sum_exp_f = sum_exp_f[:, np.newaxis]
        return exp_f / sum_exp_f

    def __softmax_entropy(self, y):
        # softmax関数を返す
        c = np.max(y)
        exp_y = np.exp(y - c)
        sum_exp_y = np.sum(exp_y)
        return exp_y / sum_exp_y

    def __GD(self, X, T):
        for i in range(300):
            # バッチアルゴリズムによる更新
            f = self.intercept_ + np.dot(X, self.coef_.T)
            z = self.__softmax(f)
            z_k = z - T
            dh1 = np.sum(z_k, axis=0)
            self.intercept_ = self.intercept_ - self.eta * dh1
            dh2 = np.dot(z_k.T, X)
            self.coef_ = self.coef_ - self.eta * dh2

    def __SGD(self, X, T):
        dh2 = np.empty((k_class, X.shape[1]))
        for j in range(3):
            np.random.seed(j)
            # np.random.shuffle(X)
            X = np.random.permutation(X)
            np.random.seed(j)
            # np.random.shuffle(y)
            T = np.random.permutation(T)
            # ミニバッチアルゴリズムによる更新
            for idx in range(0, X.shape[0]):
                f = self.intercept_ + np.dot(X[idx, :], self.coef_.T)
                z = self.__softmax_entropy(f)
                z_k = z - T[idx, :]
                dh1 = np.sum(z_k)
                self.intercept_ = self.intercept_ - self.eta * dh1
                for k in range(k_class):
                    dh2[k] = z_k[k] * X[idx, :]
                self.coef_ = self.coef_ - self.eta * dh2

    def __MSGD(self, X, T):
        batch_size = 50
        for j in range(50):
            self.eta = self.eta
            np.random.seed(j)
            np.random.shuffle(X)
            # X = np.random.permutation(X)
            np.random.seed(j)
            np.random.shuffle(T)
            # T = np.random.permutation(T)

            # ミニバッチアルゴリズムによる更新
            for idx in range(0, X.shape[0], batch_size):
                miniX = X[idx:idx + batch_size, ]
                miniT = T[idx:idx + batch_size, ]
                f = self.intercept_ + np.dot(miniX, self.coef_.T)
                z = self.__softmax(f)
                z_k = z - miniT
                dh1 = np.sum(z_k, axis=0)
                self.intercept_ = self.intercept_ - self.eta * dh1
                dh2 = np.dot(z_k.T, miniX)
                self.coef_ = self.coef_ - self.eta * dh2


if __name__ == '__main__':
    import time

    mnist = MNIST(pathMNIST='./mnist')
    k_class = 10

    # 学習データを作成
    y_train = mnist.getLabel('L')
    X_train = mnist.getImage('L')
    X_train /= 255
    X_test = mnist.getImage('T')
    X_test /= 255
    y_test = mnist.getLabel('T')

    gradient_methods = ['GD', 'SGD', 'MSGD']
    for gradient_method in gradient_methods:
        clf = Logistic_regression(gradient_method=gradient_method, seed=1)
        print(clf.eta)
        print(clf.gradient_method)
        start = time.time()
        clf.fit(X_train, y_train)
        print('学習にかかった時間:', time.time() - start)
        clf.score(X_test, y_test)
