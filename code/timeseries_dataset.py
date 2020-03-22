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


class timeseries_Dataset:

    def __init__(self, df):
        '''
        Args:
            df -> dataFrame; all raw data
        '''
        self.df = df

    @property
    def show_missing_data(self):
        na_col = {}
        for col in self.df.columns:
            na_col[col] = self.df[col].isna().sum()
        return na_col

    @property
    def show_features_name(self):
        cols = []
        for col in self.df.columns:
            cols.append(col)
        return cols

    def drop_column(self, threshold: int):
        '''
        features with number of missing data > threshold will be drop 
        '''
        na_col = self.show_missing_data
        drop_list = []

        for key, value in na_col.items():
            if value > threshold:
                drop_list.append(key)

        self.df = self.df.drop(columns=[col for col in drop_list])

    def replace_missing_data(self, mode='interpolate'):
        if mode == 'interpolate':
            self.df = self.df.reindex(fill_value='NaN').astype(
                float).interpolate(method='linear', axis=0).ffill().bfill()
        elif mode == 'mean':
            self.df = self.df.fillna(self.df.mean())
        elif mode == 'median':
            self.df = self.df.fillna(self.df.mean())

    def to_x_y(self, x_name, y_name):
        '''
        set arrtibute use to desirable columns

        Args:
            x_name -> list of str; column name
            y_name -> list of str; column name

        Returns:
            self.x -> dataFrame
            self.y -> dataFrame
        '''
        self.x = self.df.drop(columns=y_name)
        self.y = self.df[y_name]
        return self

    def trian_test_split(self, test_size=0.2, shuffle=False):
        # for sequential data we should not shuffle
        self.X_train, self.X_test, self.y_train, self.y_test = sk_ModelSelection.train_test_split(self.x, self.y,
                                                                                                  test_size=test_size, shuffle=shuffle)
        return self

    def scale_X_train(self):
        col = self.X_train.columns
        scaler = sklearn.preprocessing.StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_train = pd.DataFrame(data=self.X_train, columns=col)

    def reset(self, X_train, X_test):
        '''
        this function to re-assign X_train and X_test after being selected by XGBoost
        '''
        self.X_train = X_train
        self.X_test = X_test

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
