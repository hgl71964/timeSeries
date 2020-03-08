import time
import torch
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection as sk_ModelSelection
import seaborn as sns
import os
import sklearn
import random

'''
This script formats the time series data
'''


class timeseries_Dataset():
    def __init__(self, x, y):
        '''
        Args:
            input: all x,y after initial pre-processing type = pandas.DataFrame
        '''
        self.x = x
        self.y = y

    def trian_test_split(self, test_size=0.2, shuffle=False):
        self.X_train, self.X_test, self.y_train, self.y_test = sk_ModelSelection.train_test_split(self.x, self.y,
                                                                                                  test_size=test_size, shuffle=shuffle)

    def scale_X_train(self):
        col = self.X_train.columns
        scaler = sklearn.preprocessing.StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_train = pd.DataFrame(data=self.X_train, columns=col)

    def to_tensor(self):
        self.X_train = torch.from_numpy(self.X_train.values).float()
        self.y_train = torch.from_numpy(self.y_train.values).float()
        self.X_test = torch.from_numpy(self.X_test.values).float()
        self.y_test = torch.from_numpy(self.y_test.values).float()

    def to_numpy(self):
        self.X_train = self.X_train.values
        self.y_train = self.y_train.values
        self.X_test = self.X_test.values
        self.y_test = self.y_test.values

    @staticmethod
    def estimated_autocorrelation(x):
        """
        http://stackoverflow.com/q/14297012/190597
        http://en.wikipedia.org/wiki/Autocorrelation#Estimation

        To estimate the autocorrelation function of x,
            in order to determine the encode sequence length
        Args: 
            x, type pandas.Series, 1-d series
        """
        x = x.values  # to 1-d array
        n = len(x)
        variance = x.var()
        x = x-x.mean()
        r = np.correlate(x, x, mode='full')[-n:]
        assert np.allclose(r, np.array(
            [(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
        result = r/(variance*(np.arange(n, 0, -1)))
        return result
