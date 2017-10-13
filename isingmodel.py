#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 19:54:04 2017

@author: yohei
"""
import numpy as np
import pandas as pd


class Isingmodel(object):
    """イジングモデルの画像データを作るクラス

    パラメータ
    ---------------------
    size : int
            画像（正方形）の縦横サイズ
    n_samples : int
            データ数
    h : float
            磁場
    tem : float
             絶対温度
    j : float
            正の定数
    n_iter : int
            イテレーション回数
    属性
    ---------------------
    X_ : array-like, shape = (n_samples, 画素数)
        データ行列

    """

    def __init__(self, size=50, n_samples=1000, h=0, j=1, temp=10, n_iter=5):
        self.size = size
        self.n_samples = n_samples
        self.h = h
        self.temp = temp
        self.j = j
        self.n_iter = n_iter

    def run_gibbs_sampling(self):
        """
        Gibbsサンプリングを行うメソッド．
        中で任意の原子についてスピンさせる spin_directionメッソドを呼ぶ．
        """
        Data = pd.DataFrame(np.random.randint(2, size=(
            self.n_samples, self.size * self.size)) * 2 - 1)  # -1(下),1(上) でランダムに初期化
        self.X_ = Data
        for i, data in enumerate(Data.values):  # 行ごとにループ
            data = data.reshape(self.size, self.size)  # 1つのデータを画像のように縦横に変形
            for _ in range(self.n_iter):
                lattice = pd.DataFrame([(y, x) for x in range(self.size)
                                        for y in range(self.size)])
                lattice.reindex(np.random.permutation(
                    lattice.index))  # ランダムに並べ替え
                for x, y in lattice.values:
                    data[x, y] = self.spin_direction(data, x, y)
            self.X_.iloc[i, :] = data.reshape(1, -1)

        return self.X_

    def spin_direction(self, data, x, y):
        """
        ある原子にdata[x, y] に注目してその周りの原子をみてスピンさせるかどうかを決める
        """
        energy = self.h
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # Cyclic boundary condition
            if x + dx < 0:
                dx += self.size
            if y + dy < 0:
                dy += self.size
            if x + dx >= self.size:
                dx -= self.size
            if y + dy >= self.size:
                dy -= self.size
            energy += self.j * data[x + dx, y + dy]  # エネルギーを求める

        if self.temp == 0:
            p = (np.sign(energy) + 1) * 0.5  # 絶対零度では周りのスピン方向に向きを合わせる
        else:
            p = 1 / (1 + np.exp(-2 * (1 / self.temp) * energy))
        if np.random.rand() <= p:
            spin = 1
        else:
            spin = -1
        return spin
