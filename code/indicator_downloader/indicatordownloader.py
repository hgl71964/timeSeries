import numpy as np
import json
import pandas as pd
import requests

from datetime import date, datetime
from binance.client import Client


def merge(quandl_api_key, GLASSNODE_API_KEY, fullpath):
  merged_feats= quandlDownloader(quandl_api_key)
  bitfinex = bitfinexDownload(fullpath)
  bitfinex['Date'] = pd.to_datetime(bitfinex['Date'])
  #merge quandl with bitfinex
  merged_feats = pd.merge(merged_feats,bitfinex,how='left',on='Date')
  bitmex = cleanBitMex(bitmexDownload(fullpath, write=True))
  #merge with bitmex
  merged_feats = pd.merge(merged_feats,bitmex, how='left',on='Date')
  #merge with sp500
  sp500_data = sp500()
  merged_feats = pd.merge(merged_feats,sp500_data, how='left',on='Date')
  glassnode = glassnodeDownloader(GLASSNODE_API_KEY, fullpath)
  merged_feats = pd.merge(merged_feats,glassnode, how='left',on='Date')
  return merged_feats