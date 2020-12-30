from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import shap
import os


binanceFeats = ['BTC_volume', 'BTC_quote_asset_volume', 'BTC_number_of_trades',
                'BTC_ taker_buy_base_asset_volume', 'BTC_taker_buy_quote_asset_volume', 'BTC_returns']

rivalFeats = ['ETH_returns', 'snp_returns', 'LINK_returns',
              'XTZ_returns', 'BNB_returns', 'EOS_returns']

onchainFeats = ['DIFF', 'Hashrate_Price_Multiple', 'Mining Revenue', ]

bitfinexFeats = ['longs_mean', 'shorts_mean', 'longs_max', 'shorts_max', 'longs_min',
                 'shorts_min', 'longs_close', 'shorts_close', 'longs_open', 'shorts_open']

bitmexFeats = ['fundingRateClose', 'fundingRateDailyClose', 'fundingRateOpen', 'fundingRateDailyOpen', 'fundingRateMean',
               'fundingRateDailyMean', 'openInterestClose', 'openValueClose', 'openInterestOpen', 'openValueOpen', 'openInterestMean', 'openValueMean']

glassnodeFeats = ['exchange_outflow', 'exchange_inflow',
                  'stablecoin_supply_ratio', 'sopr', 'net_unrealized_profit_loss', 'mvrv']

technicalFeats = ['BTC_ma_21_rate', 'BTC_ma_21_std_center',
                  'BTC_ma_21_std', 'BTC_EMA_Cross', 'BTC_mayer_multiple']

feats = binanceFeats + rivalFeats + onchainFeats + \
    bitfinexFeats + bitmexFeats + glassnodeFeats + technicalFeats


def selectTrainingFeatures(data2, fullpath2, TEST_SIZE, TRANSACTION_COSTS=0,param='direction',printFeats=False, write=True):
    X = data2[feats]
    if write == True:
        X.to_csv(f"{fullpath2}/train.csv", index=False)
    if printFeats == True:
        print(feats)
    # if param == 'direction'
    y = X['BTC_returns'].shift(-1) > (1 + TRANSACTION_COSTS)
    X_train, X_test, y_train, y_test =  train_test_split(X.iloc[:-1], y.iloc[:-1], test_size=TEST_SIZE,shuffle=False)        
    return X_train, X_test, y_train, y_test


def diagnosis(y_actual, y_pred_proba):
    print(f"ACCURACY: {accuracy_score(y_actual, np.round(y_pred_proba))}")
    print(f"ROC AUC Score: {roc_auc_score(y_actual, y_pred_proba)}")
    return


def autofit(X_train, X_test, y_train, y_test, c):

    cbc = CatBoostClassifier(n_estimators=500,
                             max_depth=5,
                             thread_count=10,
                             verbose=0)
    trainPool = Pool(X_train, y_train, feature_names=list(
        X_test.columns), thread_count=1)
    cbc.fit(trainPool)

    xgbc = XGBClassifier(n_estimators=500,
                         max_depth=5,
                         objective='binary:logistic')
    xgbc.fit(X_train, y_train)
    print("Model Predictive Performance on Backtest Period")
    print("--------------------------")
    print("XGBOOST:")
    diagnosis(y_test.values * 1, xgbc.predict_proba(X_test)[:, 1])
    print("Catboost")
    diagnosis(y_test.values * 1, cbc.predict_proba(X_test)[:, 1])
    models = {}
    models['cbc'] = cbc
    models['xgbc'] = xgbc

    featImportances = dict(sorted(
        zip(X_train.columns, models['cbc'].get_feature_importance()), key=lambda k: k[1]))

    fig, ax = plt.subplots(figsize=(10, 30), nrows=3)
    ax[0].barh(list(featImportances.keys()), list(featImportances.values()),)

    xgb.plot_importance(models['xgbc'], ax=ax[1])

    train_df = pd.DataFrame(models['cbc'].predict_proba(X_train)[:, 1])
    train_df['xgbc'] = models['xgbc'].predict_proba(X_train)[:, 1]
    train_df.columns = list(models.keys())
    test_df = pd.DataFrame(models['cbc'].predict_proba(X_test)[:, 1])
    test_df['xgbc'] = pd.DataFrame(models['xgbc'].predict_proba(X_test)[:, 1])
    test_df.columns = list(models.keys())

    # models['ensemble'] = CatBoostClassifier(n_estimators=500,
    #                         max_depth=2,verbose=0)
    models['ensemble'] = LogisticRegression()
    models['ensemble'].fit(train_df, y_train)

    featImportances = dict(sorted(
        zip(list(models.keys()), models['ensemble'].coef_[0]), key=lambda k: k[1]))
    ax[2].barh(list(featImportances.keys()), list(featImportances.values()),)

    print("ENSEMBLE")
    diagnosis(y_test, models['ensemble'].predict_proba(test_df)[:, 1])
    return models, ax


# this plots predicted vs actual
def plotPredictions(models, data2, X_test):
    last_obs = X_test.index[0]
    dates = data2.loc[last_obs:]['Date']
    fig = go.Figure()
    for x in models.keys():
        if x != "ensemble":
            fig.add_trace(go.Scatter(x=dates, y=models[x].predict_proba(
                data2.loc[last_obs:, feats])[:, 1], mode='lines', name=x))

    test_df = pd.DataFrame(models['cbc'].predict_proba(
        data2.loc[last_obs:, feats])[:, 1])
    test_df['xgbc'] = pd.DataFrame(
        models['xgbc'].predict_proba(data2.loc[last_obs:, feats])[:, 1])
    test_df.columns = ['cbc', 'xgbc']

    fig.add_trace(go.Scatter(x=dates, y=models['ensemble'].predict_proba(
        test_df)[:, 1], mode='lines', name="Ensemble"))
    ys = (((data2.loc[last_obs:, 'BTC_returns'] /
            data2.loc[last_obs:, 'BTC_returns'].shift(-1)).iloc[:-1]) > 1) * 1
    fig.add_trace(go.Scatter(x=dates.iloc[:-1],
                             y=ys,
                             mode='markers',
                             name="Actual Direction",
                             marker_color=1-ys,
                             marker_symbol=ys*3,
                             marker_line_width=1,
                             marker={"size": 12, "colorscale": "Bluered"}))
    fig.update_layout(title="Model Predictive Performance for Backtest Period")
    return fig


def recommendTrade(data2, X, models):
    date = data2.iloc[-1]['Date']
    print(f"Trade Recommendation: {date}\n--------------------------")
    X2 = pd.DataFrame(models['cbc'].predict_proba(X)[:, 1])
    print(f"Catboost Prediction p: {X2.iloc[:,0].values[0]}")
    X2['xgbc'] = models['xgbc'].predict_proba(X)[:, 1]
    print(f"XGBoost Prediction p: {X2.iloc[:,1].values[0]}")
    X2.columns = ['cbc', 'xgbc']
    p = models['ensemble'].predict_proba(X2)[:, 1]
    proportion = 2 * (p - 0.5) * 100
    print("Ensemble p : ", p[0])
    print("--------------------------")
    if p < 0.5:
        print(
            f"The model recommends a short : {abs(proportion[0]):.3f}% of capital")
    else:
        print(
            f"The model recommends a long : {abs(proportion[0]):.3f}% of capital")
    print("--------------------------")
    return
