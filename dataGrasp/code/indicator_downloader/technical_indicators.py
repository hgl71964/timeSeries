import numpy as np
import json
import pandas as pd


#add extra preprocessing here
def transform(df, ti=False):
	df2 = df.copy()
	df2['ohlc'] = np.mean(df2[['open', 'high', 'low','close']].values,axis=1)
	df2['returns'] = (df2['close'] / df2['close'].shift(1))
	df2['date'] = pd.to_datetime(df2['open_time']).dt.date  
	return df2


def addTechnicalIndicators(df, ticker):
	df2 = df.copy()
	MA_WINDOW = 21
	df2[ticker+'_ma_21_rate'] = (df2[ticker+'_close'] /  df2[ticker+'_close'].rolling(MA_WINDOW).mean()) - 1
	df2[ticker+'_ma_21_std_center'] = df2[ticker+'_ma_21_rate'].rolling(MA_WINDOW * 2).mean()
	df2[ticker+'_ma_21_std'] = df2[ticker+'_ma_21_rate'].rolling(MA_WINDOW * 2).std()
	
	#EMA cross	
	df2[ticker+'_EMA_Cross'] = df2[ticker+'_close'].ewm(span=25).mean() > df2[ticker+'_close'].ewm(span=50).mean()


	#Mayer multiple
	df2[ticker+'_mayer_multiple'] = df2[ticker+'_close'] / df2[ticker+'_close'].rolling(200).mean()
	return df2