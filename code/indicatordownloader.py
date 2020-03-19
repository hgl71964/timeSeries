import numpy as np
import json
import pandas as pd
import requests

from datetime import date, datetime
from binance.client import Client

#Read in any data if it already exists

def bitfinexDownload(fullpath, write=False):
  long_success=True
  short_success=True
  try:
    bitfinex_longs = pd.read_csv(f"{fullpath}/bitfinex_longs.csv")  
    bitfinex_shorts = pd.read_csv(f"{fullpath}/bitfinex_shorts.csv")
    bitfinex_longs.columns=['timestamp','longs']
    bitfinex_shorts.columns=['timestamp','shorts']  
  except:
    long_success=False
    short_success=False
  #keep downloading until you see some data that is already in the table
  r = requests.get("https://api-pub.bitfinex.com/v2/stats1/pos.size:1m:tBTCUSD:long/hist?limit=10000")
  longs = r.json()

  r = requests.get("https://api-pub.bitfinex.com/v2/stats1/pos.size:1m:tBTCUSD:short/hist?limit=10000")
  shorts = r.json()
  if long_success == True:
    while longs[-1][0] > bitfinex_longs.iloc[0]['timestamp']:
      end = longs[-1][0]
      r = requests.get(f"https://api-pub.bitfinex.com/v2/stats1/pos.size:1m:tBTCUSD:long/hist?limit=10000&end={end}")
      longs += r.json()
  if short_success == True:
    while shorts[-1][0] > bitfinex_longs.iloc[0]['timestamp']:
      end = shorts[-1][0]
      r = requests.get(f"https://api-pub.bitfinex.com/v2/stats1/pos.size:1m:tBTCUSD:short/hist?limit=10000&end={end}")
      shorts += r.json()

  l1 = pd.DataFrame(longs)
  l1.columns = ['timestamp','longs']
  s1 = pd.DataFrame(shorts)
  s1.columns = ['timestamp','shorts']

  if long_success==True:
  	bitfinex_longs = pd.concat([bitfinex_longs,l1],axis=0)
  else:
  	bitfinex_longs = l1
  if short_success==True:
  	bitfinex_shorts = pd.concat([bitfinex_shorts,s1],axis=0)
  else:
  	bitfinex_longs = s1
  
  if write == True:
    bitfinex_longs.drop_duplicates().to_csv(f"{fullpath}/bitfinex_longs.csv",index=False)
    bitfinex_shorts.drop_duplicates().to_csv(f"{fullpath}/bitfinex_shorts.csv",index=False)

  #The data is minute resolution so downsample to daily
  bitfinex = pd.merge(bitfinex_longs,bitfinex_shorts,on='timestamp')
  bitfinex['Date'] = bitfinex['timestamp'].apply(lambda x: datetime.fromtimestamp(x/1000))
  bitfinex['Date'] = pd.to_datetime(bitfinex['Date']).dt.date
  bitfinex = bitfinex.drop_duplicates()

  bitfinex2 = bitfinex.groupby(['Date'])[['longs','shorts']].mean().reset_index()
  bitfinex2 = pd.merge(bitfinex2, bitfinex.groupby(['Date'])[['longs','shorts']].max().reset_index(),on='Date')
  bitfinex2 =pd.merge(bitfinex2, bitfinex.groupby(['Date'])[['longs','shorts']].min().reset_index(),on='Date')
  bitfinex2 =pd.merge(bitfinex2, bitfinex.groupby(['Date'])[['longs','shorts']].nth(-1).reset_index(),on='Date')
  bitfinex2 =pd.merge(bitfinex2, bitfinex.groupby(['Date'])[['longs','shorts']].nth(0).reset_index(),on='Date')
  bitfinex2.columns = ['Date','longs_mean','shorts_mean','longs_max',
                      'shorts_max','longs_min','shorts_min',
                      'longs_close','shorts_close','longs_open','shorts_open']
  if write == True:
    bitfinex2.to_csv(f"{fullpath}/bitfinex_indicators.csv",index=False)
  return bitfinex2



def quandlDownloader(api_key):
  names = {"Bitcoin_Adjusted_Difficulty":"DIFF",
         "Hashrate_Price_Multiple":"HRATE",
         "Mining Revenue":"MIREV"}
  dfs = {}
  for x in names:
    dfs[x] = pd.read_csv(f'https://www.quandl.com/api/v3/datasets/BCHAIN/{names[x]}.csv?api_key={api_key}')
    dfs[x].columns = ['Date',x]

  merged_feats = dfs[list(names.keys())[0]]
  for x in list(names.keys())[1:]:
    merged_feats = pd.merge(dfs[x],merged_feats,left_on='Date',right_on='Date')
  merged_feats['Date'] = pd.to_datetime(merged_feats['Date'])
  return merged_feats

#3 daily observations
def bitmexDownload():
  r = requests.get('https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=500&reverse=True')
  r2 = requests.get('https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=500&start=500&reverse=true')
  funding_rate = pd.io.json.json_normalize(r2.json())
  funding_rate = pd.concat([pd.io.json.json_normalize(r.json()), funding_rate])
  funding_rate['Date'] = pd.to_datetime(pd.to_datetime(funding_rate['timestamp']).dt.date)

  funding_rate2 = funding_rate.groupby('Date')[['fundingRate','fundingRateDaily']].nth(-1).reset_index()
  funding_rate2 = pd.merge(funding_rate2, funding_rate.groupby('Date')[['fundingRate','fundingRateDaily']].nth(0).reset_index(),on='Date')
  funding_rate2 = pd.merge(funding_rate2, funding_rate.groupby('Date')[['fundingRate','fundingRateDaily']].mean().reset_index(),on='Date')
  funding_rate2.columns = ['Date','fundingRateClose','fundingRateDailyClose', 'fundingRateOpen','fundingRateDailyOpen','fundingRateMean','fundingRateDailyMean']
  return funding_rate2

#sp500 data
def sp500():
	sp500 = pd.read_csv('https://stooq.com/q/d/l/?s=^spx&i=d')
	sp500['Date'] = pd.to_datetime(sp500['Date'])
	sp500['ohlc'] = sp500.apply(lambda row: (row.Open + row.Close +\
	                                 row.High+row.Low)/4, axis = 1)
	sp500['returns'] = (sp500['ohlc'] / sp500['ohlc'].shift(1))
	sp500.iloc[0,-1] = 1
	sp500.columns = ["Date"] +["snp_"+ x for x in list(sp500.columns)[1:]]
	return sp500

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

def downloadWrapper(tickers, API_SECRET, API_KEY, FREQ, fullpath,write=False):
  prices = {}
  for x in tickers:
    if x == "BTC":
      prices[x] =  transform(downloadData(x, API_KEY, API_SECRET, FREQ, fullpath), True) #change to HOUR, MINUTE if you want
    else:
      prices[x] =  transform(downloadData(x, API_KEY, API_SECRET, FREQ, fullpath))

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
  return temp

def merge(quandl_api_key, fullpath):
  merged_feats= quandlDownloader(quandl_api_key)
  bitfinex = bitfinexDownload(fullpath)
  bitfinex['Date'] = pd.to_datetime(bitfinex['Date'])
  #merge quandl with bitfinex
  merged_feats = pd.merge(merged_feats,bitfinex,how='left',on='Date')
  bitmex = bitmexDownload()
  #merge with bitmex
  merged_feats = pd.merge(merged_feats,bitmex, how='left',on='Date')
  #merge with sp500
  sp500_data = sp500()
  merged_feats = pd.merge(merged_feats,sp500_data, how='left',on='Date')
  return merged_feats