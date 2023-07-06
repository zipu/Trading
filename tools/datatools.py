import os

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def rolling_window(arr, window):
    """
    numpy 어레이의 rolling window array를 반환함
    stride trick을 사용하기 때문에 메모리 사용에 효율적임
    ex)
    arr: [1,2,3,4,5,6,7]
    window: 3
    return: 
    [ [1,2,3],
      [2,3,4],
      [3,4,5],
      [4,5,6],
      [5,6,7] ]
    """

    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def norm(data):
    """
    Data Normalization
    머신러닝 훈련 데이터 만들때 주로 사용됨
    input: numpy array (num_dim, period)
    """
    return (data-data.mean(axis=0))/data.std(axis=0).mean()


def split(arr):
    """
    arr의 연속된 값들을 그룹으로 묶음
    ex) 
    input: [1,2,3,6,7,8,10]
    output: [[1,2,3],[6,7,8],[10]]
    """
    return np.split(arr, np.where(np.diff(arr) != 1)[0]+1)
