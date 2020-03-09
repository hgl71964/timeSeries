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
    def __init__(self, x, y):
        '''
        This class format data for xgboost
        Args:
            np.darray
            x: features, excluding return, [N_sample,N_feature]
            y: return, [N_sample,]
        '''
        if type(x) is pd.DataFrame:
            self._x = x.values
        else:
            self._x = x

        if type(y) is pd.Series:
            self._y = y.values
        else:
            self._y = y

        self.full_data = np.hstack([self._x, self._y.reshape(-1, 1)])

    def walk_forward_split(self,
                           encode_len: int,
                           pred_len: int):
        '''
        Args:
            see desirable split: https://github.com/guol1nag/datagrasp/blob/master/README.md
        '''
        N_samples = self._x.shape[0]

        for i in range(0, N_samples, pred_len):
            encode = self.full_data[i:min(
                i+encode_len, N_samples)]

            pred = self._y[i+encode_len:min(
                i+encode_len+pred_len, N_samples)]
            if pred.shape[0] != pred_len:
                break

            if i == 0:
                self.x = np.expand_dims(encode, axis=0)
                self.y = np.expand_dims(pred, axis=0)
            else:
                try:
                    self.x = np.concatenate(
                        [self.x, np.expand_dims(encode, axis=0)], axis=0)
                    self.y = np.concatenate(
                        [self.y, np.expand_dims(pred, axis=0)], axis=0)

                except:  # residual
                    pass
        print('maximum batch_size:', self.x.shape[0])

    def flatten_data(self):
        pass

    # def batcher(self, batch_size):
    #     '''
    #     Args:
    #         -> output results attributes:
    #             x  [batch_size, encode_len, N_feature]
    #             y  [batch_size, encode_len+pred_len]
    #     '''
    #     l = len(self.x)
    #     for batch in range(0, l, batch_size):
    #         yield (self.x[batch:min(batch + batch_size, l)], self.y[batch:min(batch + batch_size, l)])


if __name__ == "__main__":
    x = np.random.rand(20, 4)
    y = np.random.rand(20)
    xgb_data = xgboost_dataset(x, y)
    xgb_data.walk_forward_split(5, 2)
    for i, j in xgb_data.batcher(2):
        print(i.shape, j.shape)
