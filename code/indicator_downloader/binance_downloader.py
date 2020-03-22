import numpy as np
import json
import pandas as pd
import requests

from datetime import date, datetime
from binance.client import Client

from technical_indicators import *

#Read in any data if it already exists
def downloadData(ticker, API_KEY, API_SECRET, FREQ, fullpath, write=False):
	'''
	Inputs: Binance API_KEY and API_SECRET, Binance symbol ticker, interval (Client.KLINE_INTERVAL_1HOUR, Client.KLINE_INTERVAL_1MINUTE)
	Outputs: Writes to local
	'''
	keys = {"HOUR":Client.KLINE_INTERVAL_1HOUR, "MINUTE":Client.KLINE_INTERVAL_1MINUTE,"DAY":Client.KLINE_INTERVAL_1DAY}
	binance_interval = keys[FREQ]
	client = Client(API_KEY, API_SECRET)
	klines = client.get_historical_klines(ticker+"USDT", binance_interval, "1 Jan, 2015")

	#from binance api docs
	columns = ["open_time","open","high","low","close","volume","close_time", "quote_asset_volume", "number_of_trades", " taker_buy_base_asset_volume", "taker_buy_quote_asset_volume","ignore"]
	df = pd.DataFrame(klines)
	df.columns=columns

	#times are in milliseconds
	df['open_time'] = df['open_time'].apply(lambda x: datetime.fromtimestamp(x/1000))
	df['close_time'] = df['close_time'].apply(lambda x: datetime.fromtimestamp(x/1000))
	feats = ['open','high','low','close','volume','quote_asset_volume',' taker_buy_base_asset_volume','taker_buy_quote_asset_volume']
	df[feats] = df[feats].astype('float64')
 	
	if write == True:
		path = f"{fullpath}/{ticker}USDT_{FREQ}.csv"
		df.to_csv(path,index=False)
	return df

def downloadWrapper(tickers, API_SECRET, API_KEY, FREQ, fullpath,write=False):
  prices = {}
  for x in tickers:
    if x == "BTC":
      prices[x] =  transform(downloadData(x, API_KEY, API_SECRET, FREQ, fullpath, write)) #change to HOUR, MINUTE if you want
      print(f"DONE - {x}")
    else:
      prices[x] =  transform(downloadData(x, API_KEY, API_SECRET, FREQ, fullpath, write))
      print(f"DONE - {x}")

  #Columns to take; drop open_time, close_time
  cols = ['open', 'high', 'low', 'close', 'volume','quote_asset_volume', 'number_of_trades',
        ' taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume','ohlc', 'returns']
  temp = prices[tickers[0]][['date']+cols]
  temp.columns = ['date']+[tickers[0]+"_"+ col for col in cols]
  #Merge all price datas
  for x in tickers[1:]:
    temp2 = pd.DataFrame(prices[x][['date']+cols])
    temp2.columns = ['date']+[x +"_" + col for col in cols]
    temp = pd.merge(temp, temp2,how='left',on='date')
  temp['date'] = pd.to_datetime(temp['date'])
  if write == True:
  	temp.to_csv(f"{fullpath}/UNIVERSE_{FREQ}.csv",index=False)
  print(f"DOWNLOADED ALL ASSETS")
  return temp


