import numpy as np
import json
import pandas as pd


#add extra preprocessing here
def transform(df, ti=False):
	df2 = df.copy()
	df2['ohlc'] = np.mean(df2[['open', 'high', 'low','close']].values,axis=1)
	df2['returns'] = (df2['close'] / df2['close'].shift(1))
	df2['date'] = pd.to_datetime(df2['open_time']).dt.date

	if ti == True:
		MA_WINDOW = 21
		df2['ma_21_rate'] = (df2['close'] /  df2['close'].rolling(MA_WINDOW).mean()) - 1
		df2['ma_21_std_center'] = df2['ma_21_rate'].rolling(MA_WINDOW * 2).mean()
		df2['ma_21_std'] = df2['ma_21_rate'].rolling(MA_WINDOW * 2).std()
		
		#EMA cross	
		df2['EMA_Cross'] = df2['close'].ewm(span=25).mean() > df2['close'].ewm(span=50).mean()


		#Mayer multiple
		df2['mayer_multiple'] = df2['close'] / df2['close'].rolling(200).mean()

  
	return df2