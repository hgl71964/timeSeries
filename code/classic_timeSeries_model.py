import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as sk_ModelSelection
import seaborn as sns
import os
import sklearn
import random
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA


# for dirname, _, filenames in os.walk('./data/coin_price'):
#     print('file are')
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

'''
This script implement some classical algorithms in time series forecasting, 
            such that hopufully representative data feature can be acquired 
'''


class AR_model():
    def __init__(self, data, pred_len):
        '''
        The autoregression (AR) method models the next step in the sequence 
            as a linear function of the observations at prior time steps

        Args:
            data: np.array sequential time series data
            pred_len: the length you hope to predict
        '''

        self.data = data
        self.pred_len = pred_len

    def pred(self):
        '''
        Args:
            output: -> np.array predictive data
        '''
        model = AR(self.data)
        model_fit = model.fit()
        yhat = model_fit.predict(start=len(self.data), end=len(
            self.data)+self.pred_len)  # predict the future 10 steps
        return yhat


class MA_model():
    def __init__(self, data, pred_len):
        '''
        The moving average (MA) method models the next step in the sequence as 
            a linear function of the residual errors from a mean process at prior time steps.

        Args:
            data: np.array sequential time series data
            pred_len: the length you hope to predict
        '''

        self.data = data
        self.pred_len = pred_len

    def pred(self):
        '''
        Args:
            output: -> np.array predictive data
        '''
        model = ARMA(self.data, order=(0, 1))
        model_fit = model.fit(disp=False)
        yhat = model_fit.predict(start=len(self.data), end=len(
            self.data)+self.pred_len)  # predict the future 10 steps
        return yhat


class ARMA_model():
    def __init__(self, data, pred_len):
        '''
        Args:
            data: np.array sequential time series data
            pred_len: the length you hope to predict
        '''

        self.data = data
        self.pred_len = pred_len

    def pred(self):
        '''
        Args:
            output: -> np.array predictive data
        '''

        model = ARMA(self.data, order=(2, 1))
        model_fit = model.fit(disp=False)
        yhat = model_fit.predict(start=len(self.data), end=len(
            self.data)+self.pred_len)  # predict the future 10 steps
        return yhat


class ARIMA_model():
    def __init__(self, data, pred_len):
        '''
        Args:
            data: np.array sequential time series data
            pred_len: the length you hope to predict
        '''

        self.data = data
        self.pred_len = pred_len

    def pred(self):
        '''
        Args:
            output: -> np.array predictive data
        '''

        model = ARIMA(self.data, order=(1, 1, 1))
        model_fit = model.fit(disp=False)
        yhat = model_fit.predict(start=len(self.data), end=len(
            self.data)+self.pred_len, typ='levels')  # predict the future 10 steps
        return yhat


if __name__ == "__main__":
    pass
