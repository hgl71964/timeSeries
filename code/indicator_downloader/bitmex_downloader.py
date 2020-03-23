import numpy as np
import json
import pandas as pd
import requests
import boto3

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
    print("found preexisting data in table - BITMEX")
  except:
    success = False

  r = requests.get('https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=300&reverse=True')
  funding_rate = pd.io.json.json_normalize(r.json())
  count = 300
  if success == True:
    while pd.to_datetime(funding_rate['timestamp'].iloc[-1]) > start_date:
      r = requests.get(f'https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=300&start={count}&reverse=True')
      temp = pd.io.json.json_normalize(r.json())
      funding_rate = pd.concat([funding_rate, temp],axis=0)
      count += 300
  #download 3300 rows f ~3 years of data if there is none
  else:
    for i in range(10):
      r = requests.get(f'https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=300&start={count}&reverse=True')
      temp = pd.io.json.json_normalize(r.json())
      funding_rate = pd.concat([funding_rate, temp],axis=0)
      count += 300
  funding_rate['timestamp'] = pd.to_datetime(funding_rate['timestamp'])
  if success == True:
    funding_rate2 = pd.concat([prior_df,funding_rate],axis=0,
                              sort=True).reset_index(drop=True)
    funding_rate2 = funding_rate2.drop_duplicates(['timestamp'],keep='first')                  
  else:
    funding_rate2 = funding_rate.drop_duplicates(['timestamp'],keep='first')
  if write == True:
    funding_rate2.reset_index(drop=True).to_csv(f"{fullpath}/bitmex_fundingrate.csv",index=False)
  return funding_rate2


def cleanBitMex(funding_rate):
  funding_rate['Date'] = pd.to_datetime(pd.to_datetime(funding_rate['timestamp']).dt.date)
  funding_rate2 = funding_rate.groupby('Date')[['fundingRate','fundingRateDaily']].nth(-1).reset_index()
  funding_rate2 = pd.merge(funding_rate2, funding_rate.groupby('Date')[['fundingRate','fundingRateDaily']].nth(0).reset_index(),on='Date')
  funding_rate2 = pd.merge(funding_rate2, funding_rate.groupby('Date')[['fundingRate','fundingRateDaily']].mean().reset_index(),on='Date')
  funding_rate2.columns = ['Date','fundingRateClose','fundingRateDailyClose', 'fundingRateOpen','fundingRateDailyOpen','fundingRateMean','fundingRateDailyMean']
  return funding_rate2

def bitmexDownload2(fullpath,AWS_ACCESS_KEY,AWS_SECRET_ACCESS, write=True):
  s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY,aws_secret_access_key=AWS_SECRET_ACCESS)
  s3.download_file('256-ventures', 'data/Bitmex/OI_OV_XBT.csv', 'OI_OV_XBT.csv')
  bitmex_oiov = pd.read_csv("OI_OV_XBT.csv")
  bitmex_oiov['Date'] = pd.to_datetime(pd.to_datetime(bitmex_oiov['timestamp'],utc=True).dt.date)
  bitmex_oiov2 = bitmex_oiov.groupby('Date')[['openInterest','openValue']].nth(-1).reset_index()
  bitmex_oiov2 = pd.merge(bitmex_oiov2, bitmex_oiov.groupby('Date')[['openInterest','openValue']].nth(0).reset_index(),on='Date')
  bitmex_oiov2 = pd.merge(bitmex_oiov2, bitmex_oiov.groupby('Date')[['openInterest','openValue']].mean().reset_index(),on='Date')
  bitmex_oiov2.columns = ['Date','openInterestClose','openValueClose', 'openInterestOpen','openValueOpen','openInterestMean','openValueMean']
  if write == True:
    bitmex_oiov2.to_csv(f"{fullpath}/bitmex_oiov.csv",index=False)
  return bitmex_oiov2