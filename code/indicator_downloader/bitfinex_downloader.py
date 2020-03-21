import numpy as np
import json
import pandas as pd
import requests

from datetime import date, datetime



#Read in any data if it already exists

def bitfinexDownload(fullpath, write=False):
  long_success=True
  short_success=True
  try:
    bitfinex_longs = pd.read_csv(f"{fullpath}/bitfinex_longs.csv")  
    bitfinex_shorts = pd.read_csv(f"{fullpath}/bitfinex_shorts.csv")
    bitfinex_longs.columns=['timestamp','longs']
    bitfinex_shorts.columns=['timestamp','shorts']  
    print("existing data in table")
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
  if long_success == True:
    bitfinex_longs = pd.concat([bitfinex_longs,l1],axis=0)
    bitfinex_shorts = pd.concat([bitfinex_shorts,s1],axis=0)
  else:
    bitfinex_longs = l1
    bitfinex_shorts = s1
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