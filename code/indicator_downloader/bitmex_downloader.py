import numpy as np
import json
import pandas as pd
import requests

from datetime import date, datetime

#3 daily observations
def bitmexDownload(fullpath, write=True):
  #check if there is prior data
  success = True

  #bitmex is at 8 hour intervals
  try:
    prior_df = pd.read_csv(f"{fullpath}/bitmex_fundingrate.csv")
    prior_df['timestamp'] = pd.to_datetime(prior_df['timestamp'])
    start_date = prior_df['timestamp'].iloc[0]
    print("preexisting data in table - Bitmex")
  except:
    success = False

  r = requests.get('https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=300&reverse=True')
  funding_rate = pd.json_normalize(r.json())
  count = 300
  if success == True:
    while pd.to_datetime(funding_rate['timestamp'].iloc[-1]) > start_date:
      r = requests.get(f'https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=300&start={count}&reverse=True')
      temp = pd.json_normalize(r.json())
      funding_rate = pd.concat([funding_rate, temp],axis=0)
      count += 300
  #download 3300 rows f ~3 years of data if there is none
  else:
    for i in range(10):
      r = requests.get(f'https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=300&start={count}&reverse=True')
      temp = pd.json_normalize(r.json())
      funding_rate = pd.concat([funding_rate, temp],axis=0)
      count += 300
  funding_rate['timestamp'] = pd.to_datetime(funding_rate['timestamp'])
  if success == True:
    funding_rate2 = pd.concat([prior_df,funding_rate.drop('fundingInterval',axis=1)],axis=0,
                              sort=True).reset_index(drop=True)
    funding_rate2 = funding_rate2.drop_duplicates(['timestamp'],keep='first')                  
  else:
    funding_rate2 = funding_rate.drop_duplicates(['timestamp'],keep='first')
  if write == True:
    funding_rate2.reset_index(drop=True).to_csv(f"{fullpath}/bitmex_fundingrate.csv",index=False)
  return funding_rate2.drop('fundingInterval',axis=1)


def cleanBitMex(funding_rate):
  funding_rate['Date'] = pd.to_datetime(pd.to_datetime(funding_rate['timestamp']).dt.date)
  funding_rate2 = funding_rate.groupby('Date')[['fundingRate','fundingRateDaily']].nth(-1).reset_index()
  funding_rate2 = pd.merge(funding_rate2, funding_rate.groupby('Date')[['fundingRate','fundingRateDaily']].nth(0).reset_index(),on='Date')
  funding_rate2 = pd.merge(funding_rate2, funding_rate.groupby('Date')[['fundingRate','fundingRateDaily']].mean().reset_index(),on='Date')
  funding_rate2.columns = ['Date','fundingRateClose','fundingRateDailyClose', 'fundingRateOpen','fundingRateDailyOpen','fundingRateMean','fundingRateDailyMean']
  return funding_rate2


def bitmexDownload2(fullpath, write=True):
  return