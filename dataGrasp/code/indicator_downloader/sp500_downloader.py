import numpy as np
import json
import pandas as pd
import requests

from datetime import date, datetime

def sp500(fullpath,write=True):
  now = (datetime.utcnow()-datetime(1970, 1, 1)).total_seconds()
  sp500 = pd.read_csv(f"https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=1426982400&period2={int(now)}&interval=1d&events=history")
  sp500['Date'] = pd.to_datetime(sp500['Date'])
  sp500['ohlc'] = sp500.apply(lambda row: (row.Open + row.Close +\
                                   row.High+row.Low)/4, axis = 1)
  sp500['returns'] = (sp500['Close'] / sp500['Close'].shift(1))
  sp500.loc[0,'returns'] = 1
  sp500.columns = ["Date"] +["snp_"+ x for x in list(sp500.columns)[1:]]  
  if write == True:
    sp500.to_csv(f"{fullpath}/sp500.csv",index=False)
  return sp500