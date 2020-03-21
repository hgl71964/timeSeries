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

def merge(quandl_api_key, GLASSNODE_API_KEY, fullpath):
  merged_feats= quandlDownloader(quandl_api_key, fullpath)
  print("QUANDL - COMPLETE")
  bitfinex = bitfinexDownload(fullpath)
  print("BITFINEX - COMPLETE")
  #merge quandl with bitfinex
  merged_feats['Date'] = pd.to_datetime(merged_feats['Date'])
  bitfinex['Date'] = pd.to_datetime(bitfinex['Date'])
  merged_feats = pd.merge(merged_feats,bitfinex,how='left',on='Date')
  bitmex = cleanBitMex(bitmexDownload(fullpath, write=True))
  #merge with bitmex
  merged_feats = pd.merge(merged_feats,bitmex, how='left',on='Date')
  #merge with sp500
  sp500_data = sp500(fullpath)
  print("SP500 - COMPLETE")
  merged_feats['Date'] = pd.to_datetime(merged_feats['Date'])
  sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])
  merged_feats = pd.merge(merged_feats,sp500_data, how='left',on='Date')
  glassnode = glassnodeDownloader(GLASSNODE_API_KEY, fullpath)
  glassnode['Date'] = pd.to_datetime(glassnode['Date'])
  merged_feats = pd.merge(merged_feats,glassnode, how='left',on='Date')
  print("GLASSNODE - COMPLETE")
  return merged_feats