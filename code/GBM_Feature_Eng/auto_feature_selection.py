import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as sk_ModelSelection
import seaborn as sns
import os
import sklearn
import random
import xgboost


class xgboost_utility:

    def __init__(self, X_train, y_train, X_test, y_test, feature_map, **kwargs):
        '''
        param_grid sets hyper-parameter for the model

        grid sets other parameter
        '''
        self.other_param = {

        }

        self.model_grid = {
            # Number of gradient boosted trees. Equivalent to number of boosting rounds.
            'num_estimators': kwargs['num_estimators'],

            # Maximum tree depth for base learners.
            'max_depth': kwargs['max_depth'],

            # Boosting learning rate (xgb’s “eta”)
            'learning_rate': kwargs['learning_rate'],

            # 0 (silent), 1 (warning), 2 (info), 3 (debug).
            'verbosity': kwargs['verbosity'],

            'objective': kwargs['objective'],  # regression with squared loss
            'min_child_weight': kwargs['min_child_weight'],

            # L2 norm regularization, dafault 1
            'lambda': kwargs['lambda'],
            # 'subsample': 0.5,                 # Subsample ratio of the training instance
            # 'colsample_bytree': 0.6,
            # 'reg_lambda':,                   # L2 regularization term on weights
        }
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_map = feature_map

        # instantiate model according to hyper-parameters
        self.model = xgboost.XGBRegressor(**self.model_grid)

    def training(self):
        self.model.fit(self.X_train, self.y_train,
                       eval_set=[(self.X_train, self.y_train),
                                 (self.X_test, self.y_test)],
                       eval_metric='rmse',
                       verbose=True)

    def feature_scores(self):
        raw_ranking = sorted(self.model.get_booster().get_score(
        ).items(), key=lambda x: x[1], reverse=True)
        ranking = []
        for t in raw_ranking:
            for key in self.feature_map:
                if t[0] == key:
                    ranking.append((self.feature_map[key], t[1]))
        self.ranking = ranking
        print(ranking)

    def feature_selection(self, k: int):
        '''
        Args:
            top k features that you want to preserve

        Outputs:
            return X_train, X_test with selected features
        '''

        # a list of integer
        preserve_list = []
        for i in range(k):
            for key, value in self.feature_map.items():
                if value == i[0]:
                    preserve_list.append(int(key[1:]))

        # so we only take the columns we want
        X_train = self.X_train[:sorted(preserve_list)]
        X_test = self.X_test[:sorted(preserve_list)]
        return X_train, X_test

    @staticmethod
    def show_param():
        print('''
        Instantiate

        Pass a dict contains all parameters, example is given in the following:

        all_param={
            ##### model parameters:

            'num_estimators': 1000,    #  Number of gradient boosted trees. Equivalent to number of boosting rounds.
            'max_depth': 10,            #  Maximum tree depth for base learners.
            'learning_rate': 0.3,       #  Boosting learning rate (xgb’s “eta”)
            'verbosity':1,               #  0 (silent), 1 (warning), 2 (info), 3 (debug).
            'objective': 'reg:squarederror', # regression with squared loss
            'min_child_weight': 10,  
            'lambda':1                      # L2 norm regularization, dafault 1

            ##### others:

            ''
             
            }

        ''')
