import numpy as np
import json
import pandas as pd
import requests

from datetime import date, datetime

#map feature names to QUANDL name


def quandlDownloader(QUANDL_API_KEY,fullpath, write=True):
  #check if there is prior data
  success = True
  try:
    prior_df = pd.read_csv(f"{fullpath}/quandl_indicators.csv")
    prior_df['Date'] = pd.to_datetime(prior_df['Date'])
    start_date = str(prior_df['Date'].iloc[0])[:10]
    print("preexisting data in table - Quandl")
  except:
    success = False
  #dict of quandl features and its API name     
  names = {"Bitcoin_Adjusted_Difficulty":"DIFF",
         "Hashrate_Price_Multiple":"HRATE",
         "Mining Revenue":"MIREV"}
  
  url = 'https://www.quandl.com/api/v3/datasets/BCHAIN/'

  if success == True:
    df = pd.read_csv(f'{url}{names[list(names.keys())[0]]}.csv?api_key={QUANDL_API_KEY}&start_date={start_date}')
  else:
    df = pd.read_csv(f'{url}{names[list(names.keys())[0]]}.csv?api_key={QUANDL_API_KEY}')
  df.columns = ['Date',names[list(names.keys())[0]]]
  df['Date'] = pd.to_datetime(df['Date'])

  for x in list(names.keys())[1:]:
    if success == True: 
      temp = pd.read_csv(f'https://www.quandl.com/api/v3/datasets/BCHAIN/{names[x]}.csv?api_key={QUANDL_API_KEY}&start_date={start_date}')
    else:
      temp = pd.read_csv(f'https://www.quandl.com/api/v3/datasets/BCHAIN/{names[x]}.csv?api_key={QUANDL_API_KEY}')
    temp.columns = ['Date',x]
    temp['Date'] = pd.to_datetime(temp['Date'])
    df = pd.merge(df,temp,left_on='Date',right_on='Date')
  
  df['Date'] = pd.to_datetime(df['Date'])

  #if there is existing data append
  if success == True:
    df = pd.concat([df,prior_df], axis=0).drop_duplicates()
  #if write then write to persistence
  if write == True:
    df.to_csv(f"{fullpath}/quandl_indicators.csv",index=False)
  return df