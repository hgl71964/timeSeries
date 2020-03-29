import numpy as np
import json
import pandas as pd
import requests

from datetime import date, datetime
from binance.client import Client


from bitfinex_downloader import *
from bitmex_downloader import *
from glassnode_downloader import *
from quandl_downloader import *
from sp500_downloader import *

def merge(quandl_api_key, GLASSNODE_API_KEY, AWS_ACCESS_KEY,AWS_SECRET_ACCESS, fullpath, write):
  merged_feats= quandlDownloader(quandl_api_key, fullpath, write)
  print("DONE - QUANDL")
  bitfinex = bitfinexDownload(fullpath)
  print("DONE - BITFINEX")
  #merge quandl with bitfinex
  merged_feats['Date'] = pd.to_datetime(merged_feats['Date'])
  bitfinex['Date'] = pd.to_datetime(bitfinex['Date'])
  merged_feats = pd.merge(merged_feats,bitfinex,how='left',on='Date')
  bitmex = cleanBitMex(bitmexDownload(fullpath, write=write))
  bitmex2 = bitmexDownload2(fullpath,AWS_ACCESS_KEY,AWS_SECRET_ACCESS, write=write)
  print("DONE - BITMEX")
  #merge with bitmex
  merged_feats = pd.merge(merged_feats,bitmex, how='left',on='Date')
  merged_feats = pd.merge(merged_feats,bitmex2, how='left',on='Date')
  #merge with sp500
  sp500_data = sp500(fullpath, write)
  print("DONE - SP500")
  sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])
  merged_feats = pd.merge(merged_feats,sp500_data, how='left',on='Date')
  glassnode = glassnodeDownloader(GLASSNODE_API_KEY, fullpath,write=write)
  glassnode['Date'] = pd.to_datetime(glassnode['Date'])
  merged_feats = pd.merge(merged_feats,glassnode, how='left',on='Date')
  print("DONE -GLASSNODE")
  
  merged_feats.loc[:,'snp_returns'].fillna(1,inplace=True)
  merged_feats.loc[:,'stablecoin_supply_ratio'].fillna(method='ffill',inplace=True)
  return merged_feats