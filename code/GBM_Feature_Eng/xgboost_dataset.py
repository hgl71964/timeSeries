import time
import copy
import numpy as np
import pandas as pd
import sklearn.model_selection as sk_ModelSelection
import seaborn as sns
import os
import sklearn
import random


class xgboost_dataset():
    def __init__(self, X_train, y_train, X_test, y_test):
        '''
        This class format data for xgboost
        Args:
            input type = np.darray or pandas

            X_train: features, excluding return, [N_sample,N_feature]
            y_train: return, [N_sample,]
        '''
        if type(X_train) is pd.DataFrame:
            self.X_train = X_train.values
            self.feature_map = {f'f{key}': X_train.columns[key]
                                for key in range(len(X_train.columns))}
        else:
            self.X_train = X_train

        if type(y_train) is pd.Series:
            self.y_train = y_train.values
        else:
            self.y_train = y_train

        self.full_data_train = np.hstack(
            [self.X_train, self.y_train.reshape(-1, 1)])

        if type(X_test) is pd.DataFrame:
            self.X_test = X_test.values

        else:
            self.X_test = X_test

        if type(y_test) is pd.Series:
            self.y_test = y_test.values
        else:
            self.y_test = y_test

        self.full_data_test = np.hstack(
            [self.X_test, self.y_test.reshape(-1, 1)])

    def split_dataset(self,
                      encode_len: int,
                      pred_len: int,):
        self.X_train, self.y_train = self._xgb_walk_forward_split(
            self.X_train, self.y_train, self.full_data_train, encode_len, pred_len, print_info=True)

        self.X_test, self.y_test = self._xgb_walk_forward_split(
            self.X_test, self.y_test, self.full_data_test, encode_len, pred_len, print_info=False)

    def _xgb_walk_forward_split(self,
                                x,
                                y,
                                full_data,
                                encode_len: int,
                                pred_len: int,
                                print_info):
        '''
        see desirable split: https://github.com/guol1nag/datagrasp/blob/master/README.md

        However, remember in XGBoost, data can only be formatted as X_train: [N_samples,N_features]
                                                                    y: [N_sampples,]
        It is therefore, we cannot make a batch, we can only use data in time t to predict time t+1

        Args:

            output -> np.array
                self.x [N_samples,N_features + return]
                self.y [N_samples,]
        '''
        N_samples = x.shape[0]

        for i in range(0, N_samples, pred_len):
            encode = full_data[i:min(
                i+encode_len, N_samples)]

            pred = y[i+encode_len:min(
                i+encode_len+pred_len, N_samples)]
            if pred.shape[0] != pred_len:   # to protect the prediction_length
                break

            if i == 0:
                new_x = encode.reshape(1, -1)
                new_y = pred.reshape(1, -1)
            else:
                try:
                    new_x = np.concatenate(
                        [new_x, encode.reshape(1, -1)], axis=0)
                    new_y = np.concatenate(
                        [new_y, pred.reshape(1, -1)], axis=0)
                except:  # residual
                    pass
        if print_info:
            print('number of samples:', new_x.shape[0])
        return new_x, new_y

    def batcher(self, x, y, batch_size: int):
        '''
        this method is deprecated

        Args:
            x: iterable 
            y: iterable

        Output results attributes:
            x  [batch_size, encode_len, N_feature]
            y  [batch_size, encode_len+pred_len]
        '''
        print('this method is deprecated')
        l = len(x)
        for batch in range(0, l, batch_size):
            yield (x[batch:min(batch + batch_size, l)], y[batch:min(batch + batch_size, l)])


# if __name__ == "__main__":
#     x = np.random.rand(20, 4)
#     y = np.random.rand(20)
#     xgb_data = xgboost_dataset(x, y)
#     xgb_data.xgb_walk_forward_split(1, 1)
#     print(xgb_data.x.shape)
