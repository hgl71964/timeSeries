from binance_downloader import *
from merge import merge
from bitfinex_downloader import *
from bitmex_downloader import *
from glassnode_downloader import *
from quandl_downloader import *
from sp500_downloader import *
from technical_indicators import *
import os


def getLatestData(tickers, API_SECRET, API_KEY, QUANDL_API, GLASSNODE_API_KEY, AWS_ACCESS_KEY, AWS_SECRET_ACCESS, FREQ, fullpath, fullpath2, write=True):
    '''
    here assume we have mounted google drive, and create folder to download our data
    '''
    if not os.path.exists("drive/My Drive/datagrasp"):
        os.mkdir('drive/My Drive/datagrasp')
    if not os.path.exists(os.path.join("drive/My Drive/datagrasp", "CoinPrice")):
        os.mkdir(os.path.join("drive/My Drive/datagrasp", "CoinPrice"))
    if not os.path.exists(os.path.join("drive/My Drive/datagrasp", "Indicator")):
        os.mkdir(os.path.join("drive/My Drive/datagrasp", "Indicator"))

    data = downloadWrapper(tickers, API_SECRET, API_KEY,
                           FREQ, fullpath, write=write)
    data['date'] = pd.to_datetime(data['date'])
    indicators = merge(QUANDL_API, GLASSNODE_API_KEY,
                       AWS_ACCESS_KEY, AWS_SECRET_ACCESS, fullpath2, write)
    indicators['Date'] = pd.to_datetime(indicators['Date'])
    merged_df = pd.merge(data, indicators, how='inner',
                         left_on='date', right_on='Date')
    print("MERGED INDICATORS WITH PRICE DATA")
    merged_df = addTechnicalIndicators(merged_df, "BTC")
    print("ADDED TECHNICAL INDICATORS")
    if write == True:
        merged_df.to_csv(
            f"{fullpath}/BTCUSDT_Daily_WITH_INDICATORS.csv", index=False)
    return merged_df
