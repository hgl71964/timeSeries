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
import shapley



def diagnosis(y_actual, y_pred_proba):
	print(f"ACCURACY: {accuracy_score(y_test, y_pred_proba > 0.5)}")
	print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba > 0.5)}")
	return

def autofit(X_train, X_test, y_train, y_test, c):

	cbc = CatBoostClassifier(n_estimators=500, 
	                        max_depth=8,
	                        verbose=0)
	pool = Pool(X_train, y_train, feature_names=list(X_test.columns))
	cbc.fit(pool)

	xgbc = XGBClassifier(n_estimators=500,
	                    max_depth=3,
	                    objective='reg:squarederror')
	xgbc.fit(X_train, y_train)	
	print("XGBOOST:")
	diagnosis(y_test, xgbc.predict_proba(X_test)[:,1])
	print("Catboost")
	diagnosis(y_test, cbr.predict_proba(X_test)[:,1])
	models = {}
	models['cbc'] = cbc
	models['xgbc'] = xgbc
	return models
