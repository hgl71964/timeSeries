from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, roc_auc_score, accuracy_score
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



def diagnosis(y_actual, y_pred_proba):
	print(f"ACCURACY: {accuracy_score(y_actual, y_pred_proba > 0.5)}")
	print(f"ROC AUC Score: {roc_auc_score(y_actual, y_pred_proba > 0.5)}")
	return

def autofit(X_train, X_test, y_train, y_test, c):

	cbc = CatBoostClassifier(n_estimators=1000, 
	                        max_depth=8,
	                        verbose=0)
	pool = Pool(X_train, y_train, feature_names=list(X_test.columns))
	cbc.fit(pool)

	xgbc = XGBClassifier(n_estimators=1000,
	                    max_depth=8,
	                    objective='reg:squarederror')
	xgbc.fit(X_train, y_train)	
	print("XGBOOST:")
	diagnosis(y_test, xgbc.predict_proba(X_test)[:,1])
	print("Catboost")
	diagnosis(y_test, cbc.predict_proba(X_test)[:,1])
	models = {}
	models['cbc'] = cbc
	models['xgbc'] = xgbc


	train_df = pd.DataFrame(models['cbc'].predict_proba(X_train)[:,1])
	train_df['xgbc'] = models['xgbc'].predict_proba(X_train)[:,1]
	train_df.columns = list(models.keys())
	test_df = pd.DataFrame(models['cbc'].predict_proba(X_test)[:,1])
	test_df['xgbc'] = pd.DataFrame(models['xgbc'].predict_proba(X_test)[:,1])
	test_df.columns = list(models.keys())

	models['ensemble'] = XGBClassifier(n_estimators=500,max_depth=3)
	models['ensemble'].fit(train_df, y_train)

	print("ENSEMBLE")
	diagnosis(y_test, models['ensemble'].predict_proba(test_df)[:,1])
	return models



def featureImportances(models, X_train):
  featImportances = dict(sorted(zip(X_train.columns,models['cbc'].get_feature_importance()),key=lambda k: k[1]))

  fig, ax = plt.subplots(figsize=(15,20),nrows=2)
  ax[0].barh(list(featImportances.keys()), list(featImportances.values()),)

  xgb.plot_importance(models['xgbc'],ax=ax[1])

  xgb.plot_tree(models['xgbc'],ax=ax[2])
  return ax


#this plots predicted vs actual
def plotPredictions(dates, models, X, y, last_obs):
  fig = go.Figure()
  for x in models.keys():
	  fig.add_trace(go.Scatter(x=dates, y=models[x].predict_proba(X[last_obs:])[:,1],mode='lines',name=x))
  return fig


