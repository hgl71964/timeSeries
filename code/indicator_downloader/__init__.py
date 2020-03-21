from binance_downloader import *
from merge import merge

from bitfinex_downloader import *
from bitmex_downloader import *
from glassnode_downloader import *
from quandl_downloader import *
from sp500_downloader import *

if __name__ == '__main__':
	#KEY Parameters
	API_KEY = "w1Ir6bji5wJAQgE2UqwIoH2Paa12Rup3Ropqjo9QXQiCdN0eI6VBM5MkQMNgZ40a"
	API_SECRET="IlCdbVGmIwA9yuEfmo71TZYQKR1w8VwraOPeWWZq7DT7jkvzi0uoMrSRZwquYyFd"
	QUANDL_API = 'RxRK_efya6Xyqz3iqf9W'
	GLASSNODE_API_KEY = 'cc8e1561-2dda-43c9-9fd2-61ec92900bbd'

	#change fullpath to the location of your dataset
	path = 'drive/My Drive/Datagrasp - 256 Ventures/datasets/' 
	fullpath = path+'CoinPrice'
	fullpath2 = path+"Indicator"

	tickers = ['BTC','ETH','BNB','EOS','XRP', "XTZ", "LINK"]
	FREQ = "DAY"

	#Function call
	'''
	Takes in a list of TICKERS
	your BINANCE API keys
	the frequency of your Data (DAY, HOUR, MINUTE)
	FULLPATH is the location you want to save to
	WRITE is whether you want to output to a csv
	'''
	data = downloadWrapper(tickers, API_SECRET, API_KEY, FREQ, fullpath)
	indicators = merge(QUANDL_API, GLASSNODE_API_KEY, fullpath2)
