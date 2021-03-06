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
import catboost
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error


class gradient_boost_utility:

    def __init__(self, X_train, y_train, X_test, y_test, **kwargs):
        '''
        param_grid sets hyper-parameter for the model

        grid sets other parameter

        Args:
            X_train, y_train, X_test, y_test -> np.darray
        '''
        try:
            self.xgb_param = {
                # Number of gradient boosted trees. Equivalent to number of boosting rounds.
                'num_estimators': kwargs['num_estimators'],

                # Maximum tree depth for base learners.
                'max_depth': kwargs['max_depth'],

                # Boosting learning rate (xgb’s “eta”)
                'learning_rate': kwargs['learning_rate'],

                # 0 (silent), 1 (warning), 2 (info), 3 (debug).
                'verbosity': kwargs['verbosity'],

                # regression with squared loss
                'objective': kwargs['objective'],

                'min_child_weight': kwargs['min_child_weight'],

                # L2 norm regularization, dafault 1
                'lambda': kwargs['lambda'],
                # 'subsample': 0.5,                 # Subsample ratio of the training instance
                # 'colsample_bytree': 0.6,
                # 'reg_lambda':,                   # L2 regularization term on weights
            }
            self.cat_param = {
                'n_estimators': kwargs['cat_n_estimators'],
                'max_depth': kwargs['cat_max_depth'],
                'verbose': kwargs['cat_verbose'],
            }
        except:
            print('''hyper-parameter setting fails, 
                model uses default settings''')
            self.default_model_setting

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # instantiate model according to hyper-parameters
        self.XGBoost = xgboost.XGBRegressor(**self.xgb_param)
        self.Catboost = catboost.CatBoostRegressor(**self.cat_param)

    @property
    def default_model_setting(self):
        self.xgb_param = {
            # Number of gradient boosted trees. Equivalent to number of boosting rounds.
            'num_estimators': 1000,
            'max_depth': 10,  # Maximum tree depth for base learners.
            'learning_rate': 0.3,  # Boosting learning rate (xgb’s “eta”)

            # 0 (silent), 1 (warning), 2 (info), 3 (debug).
            'verbosity': 0,

            'objective': 'reg:squarederror',  # regression with squared loss
            'min_child_weight': 10,
            'lambda': 1,                      # L2 norm regularization, dafault 1
            # 'subsample': 0.5,                 # Subsample ratio of the training instance
            # 'colsample_bytree': 0.6,
            # 'reg_lambda':,                   # L2 regularization term on weights
        }
        self.cat_param = {
            'n_estimators': 1000,
            'max_depth': 5,
            'verbose': 0,
        }

    def grid_search(self,  search):
        '''
        Args:
            search -> boolean
        '''
        if search:
            print('Auto search for hyper-parameters')
            param_grid = {'num_estimators': [500, 1000, 2000],
                          'max_depth': [5, 10, 20],
                          'learning_rate': [1e-3, 1e-1, 0.3],
                          'min_child_weight': [5, 10, 15],
                          'n_estimators': [500, 1000, 2000],
                          'cat_max_depth': [5, 10, 20],
                          }
            best_loss = float('inf')
            best_grid = {}

            for one_search in list(ParameterGrid(param_grid)):
                for key in one_search:
                    if key in self.xgb_param:
                        self.xgb_param[key] = one_search[key]
                    elif key in self.cat_param:
                        self.cat_param[key] = one_search[key]

                self.cat_param['max_depth'] = one_search['cat_max_depth']

                # instantiate
                self.XGBoost = xgboost.XGBRegressor(**self.xgb_param)
                self.Catboost = catboost.CatBoostRegressor(**self.cat_param)

                self.training()

                y_pred = self.XGBoost.predict(self.X_test)
                loss = mean_squared_error(self.y_test, y_pred)

                y_pred = self.Catboost.predict(self.X_test)
                loss += mean_squared_error(self.y_test, y_pred)

                if loss < best_loss:
                    best_loss = loss
                    best_grid = one_search
                    max_depth = one_search['cat_max_depth']

            for key in best_grid:
                if key in self.xgb_param:
                    self.xgb_param[key] = best_grid[key]
                elif key in self.cat_param:
                    self.cat_param[key] = best_grid[key]
            self.cat_param['max_depth'] = max_depth

    def training(self):
        '''
        train both GBM 
        '''
        self.XGBoost.fit(self.X_train, self.y_train,
                         eval_set=[(self.X_train, self.y_train),
                                   (self.X_test, self.y_test)],
                         eval_metric='rmse',
                         verbose=self.xgb_param['verbosity'])

        trainPool = catboost.Pool(self.X_train, self.y_train)
        self.Catboost.fit(trainPool)

    @property
    def feature_scores(self):
        '''
        xgboost features score
        '''
        raw_ranking = sorted(self.XGBoost.get_booster().get_score(
        ).items(), key=lambda x: x[1], reverse=True)

        maximum1 = raw_ranking[0][1]

        # normalisation & to dict
        self.xgb_rank = {}
        for item in raw_ranking:
            self.xgb_rank[item[0]] = item[1]

        '''
        catboost features score
        '''
        self.cat_rank = dict(sorted(zip(self.X_train.columns,
                                        self.Catboost.get_feature_importance()), key=lambda k: k[1]))
        # get largest
        maximum2 = 0
        for val in self.cat_rank.values():
            if val > maximum2:
                maximum2 = val

        '''
        combine & normalisation 
        '''
        self.norm_ranking = {key: 0 for key in self.xgb_rank}
        for key in self.norm_ranking:
            self.norm_ranking[key] = (
                self.cat_rank[key] / maximum2) + (self.xgb_rank[key] / maximum1)

        return self.norm_ranking

    def feature_selection(self, k: int):
        '''
        Args:
            top k features that you want to preserve

        Returns:
            -> list of str of feature names (which is going to preserve)
        '''

        if k+1 > len(self.norm_ranking.keys()):
            raise ValueError('you are selecting all features!')

        rank_list = sorted(
            [(key, values) for key, values in self.norm_ranking.items()], key=lambda x: x[1], reverse=True)

        preserve_list = []
        for i in range(k+1):  # add 1 because we have already included y
            preserve_list.append(rank_list[i][0])

        return preserve_list

    def prediction(self, x):
        return self.XGBoost.predict(x), self.Catboost.predict(x)

    @staticmethod
    def show_param():
        print('''
        Instantiate

        Pass a dict contains all parameters, example is given in the following:

        all_param={
            ##### XGBoost model parameters:

            'num_estimators': 1000,    #  Number of gradient boosted trees. Equivalent to number of boosting rounds.
            'max_depth': 10,            #  Maximum tree depth for base learners.
            'learning_rate': 0.3,       #  Boosting learning rate (xgb’s “eta”)
            'verbosity':1,               #  0 (silent), 1 (warning), 2 (info), 3 (debug).
            'objective': 'reg:squarederror', # regression with squared loss
            'min_child_weight': 10,  
            'lambda':1                      # L2 norm regularization, dafault 1

            ##### Catboost model parameters:

            'cat_n_estimators': 1000,
            'cat_max_depth': 5,
            'cat_verbose': 0,
            }

        ''')
