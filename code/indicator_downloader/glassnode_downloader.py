import numpy as np
import json
import pandas as pd
import requests

from datetime import date, datetime

def glassnodeDownloader(GLASSNODE_API_KEY, fullpath, write=False):
  url = 'https://api.glassnode.com/v1/metrics/'

  indicators = {"exchange_outflow":"transactions/transfers_volume_from_exchanges_sum",
                "exchange_inflow":"transactions/transfers_volume_to_exchanges_sum",
                "stablecoin_supply_ratio":"indicators/ssr",
                "sopr":"indicators/sopr",
                'net_unrealized_profit_loss':"indicators/net_unrealized_profit_loss",
                "mvrv":"market/mvrv",
                }
  #check if there is preexisting data
  success = True
  try:
    prior_df = pd.read_csv(f"{fullpath}/glassnode_indicators.csv")
    start_date = prior_df['timestamp'].iloc[-1]
  except:
    success = False
  
  if success == True:
    r = requests.get(f"{url}{indicators[list(indicators.keys())[0]]}?a=BTC&s={start_date}",{'api_key':GLASSNODE_API_KEY})
  else:
    r = requests.get(f"{url}{indicators[list(indicators.keys())[0]]}?a=BTC",{'api_key':GLASSNODE_API_KEY})
  df = pd.json_normalize(r.json())
  df.columns = ['timestamp',list(indicators.keys())[0]]

  for x in list(indicators.keys())[1:]:
    if success == True:
      r = requests.get(f"{url}{indicators[x]}?a=BTC&s={start_date}",{'api_key':GLASSNODE_API_KEY})
    else:
      r = requests.get(f"{url}{indicators[list(indicators.keys())[0]]}?a=BTC",{'api_key':GLASSNODE_API_KEY})
    temp = pd.json_normalize(r.json())
    temp.columns = ['timestamp', x]
    df = pd.merge(df, temp, how='inner',left_on='timestamp',right_on='timestamp')

  df['Date'] = pd.to_datetime(df['timestamp'].apply(lambda x: datetime.fromtimestamp(x)))
  if success == True:
    df = pd.concat([df,prior_df], axis=0).drop_duplicates().reset_index(drop=True)
  if write == True:
    df.to_csv(f"{fullpath}/glassnode_indicators.csv",index=False)
  return df