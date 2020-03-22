from binance_downloader import *
from merge import merge
from bitfinex_downloader import *
from bitmex_downloader import *
from glassnode_downloader import *
from quandl_downloader import *
from sp500_downloader import *
from technical_indicators import *


def getLatestData(tickers, API_SECRET, API_KEY, QUANDL_API, GLASSNODE_API_KEY, FREQ, fullpath, fullpath2, write=True):
	data = downloadWrapper(tickers, API_SECRET, API_KEY, FREQ, fullpath, write=True)
	data['date'] = pd.to_datetime(data['date'])
	indicators = merge(QUANDL_API, GLASSNODE_API_KEY, fullpath2)
	indicators['Date'] = pd.to_datetime(indicators['Date'])
	merged_df = pd.merge(data,indicators,how='inner',left_on='date',right_on='Date')
	print("MERGED INDICATORS WITH PRICE DATA")
	merged_df = addTechnicalIndicators(merged_df,"BTC")
	print("ADDED TECHNICAL INDICATORS")
	if write == True:
		merged_df.to_csv(f"{fullpath}/BTCUSDT_Daily_WITH_INDICATORS.csv",index=False)
	return merged_df



