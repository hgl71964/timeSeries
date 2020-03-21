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
	print(f"ACCURACY: {accuracy_score(y_actual, np.round(y_pred_proba))}")
	print(f"ROC AUC Score: {roc_auc_score(y_actual, y_pred_proba)}")
	return

def autofit(X_train, X_test, y_train, y_test, c):

	cbc = CatBoostClassifier(n_estimators=500, 
	                        max_depth=5,
                          thread_count=10,
	                        verbose=0)
	trainPool = Pool(X_train, y_train, feature_names=list(X_test.columns),thread_count=1)
	cbc.fit(trainPool)

	xgbc = XGBClassifier(n_estimators=500,
	                    max_depth=5,
	                    objective='reg:squarederror')
	xgbc.fit(X_train, y_train)	
	print("XGBOOST:")
	diagnosis(y_test.values * 1, xgbc.predict_proba(X_test)[:,1])
	print("Catboost")
	diagnosis(y_test.values * 1, cbc.predict_proba(X_test)[:,1])
	models = {}
	models['cbc'] = cbc
	models['xgbc'] = xgbc

	featImportances = dict(sorted(zip(X_train.columns,models['cbc'].get_feature_importance()),key=lambda k: k[1]))

	fig, ax = plt.subplots(figsize=(20,50),nrows=3)
	ax[0].barh(list(featImportances.keys()), list(featImportances.values()),)

	xgb.plot_importance(models['xgbc'],ax=ax[1])

	xgb.plot_tree(models['xgbc'],ax=ax[2])


	train_df = pd.DataFrame(models['cbc'].predict_proba(X_train)[:,1])
	train_df['xgbc'] = models['xgbc'].predict_proba(X_train)[:,1]
	train_df.columns = list(models.keys())
	test_df = pd.DataFrame(models['cbc'].predict_proba(X_test)[:,1])
	test_df['xgbc'] = pd.DataFrame(models['xgbc'].predict_proba(X_test)[:,1])
	test_df.columns = list(models.keys())

	models['ensemble'] = XGBClassifier(n_estimators=500,max_depth=5)
	models['ensemble'].fit(train_df, y_train)

	print("ENSEMBLE")
	diagnosis(y_test, models['ensemble'].predict_proba(test_df)[:,1])
	return models, ax






#this plots predicted vs actual
def plotPredictions(dates, models, X, y, last_obs):
  fig = go.Figure()
  for x in models.keys():
    if x != "ensemble":
        fig.add_trace(go.Scatter(x=dates, y=models[x].predict_proba(X[last_obs:])[:,1],mode='lines',name=x))
  
  test_df = pd.DataFrame(models['cbc'].predict_proba(X[last_obs:])[:,1])
  test_df['xgbc'] = pd.DataFrame(models['xgbc'].predict_proba(X[last_obs:])[:,1]) 
  test_df.columns = ['cbc','xgbc']

  fig.add_trace(go.Scatter(x=dates, y=models['ensemble'].predict_proba(test_df)[:,1],mode='lines',name="Ensemble"))  

  fig.add_trace(go.Scatter(x=dates.iloc[:-1], 
                           y=(y[last_obs:]*1),
                           mode='markers',
                           name="Actual Direction",
                           marker_color=1-(y[last_obs:]*1),
                           marker_symbol=y[last_obs:]*3,
                           marker_line_width=1,
                           marker={"size":12,"colorscale":"Bluered"}))
  return fig


def recommendTrade(date, X, models):
  print(f"{date}\n--------------------------")
  X2 = pd.DataFrame(models['cbc'].predict_proba(X)[:,1])
  print(f"Catboost Prediction p: {X2.iloc[:,0].values[0]}")
  X2['xgbc'] = models['xgbc'].predict_proba(X)[:,1]
  print(f"XGBoost Prediction p: {X2.iloc[:,1].values[0]}")
  X2.columns = ['cbc','xgbc']
  p = models['ensemble'].predict_proba(X2)[:,1]
  proportion = 2 * (p - 0.5) * 100
  print("Ensemble p : ", p[0])
  if p < 0.5:
    print(f"The model recommends a short : {abs(proportion[0]):.3f}% of capital")
  else:
    print(f"The model recommends a long : {abs(proportion[0]):.3f}% of capital")



