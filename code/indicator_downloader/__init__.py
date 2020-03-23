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



binanceFeats = ['BTC_volume', 'BTC_quote_asset_volume', 'BTC_number_of_trades','BTC_ taker_buy_base_asset_volume', 'BTC_taker_buy_quote_asset_volume','BTC_returns']

rivalFeats = ['ETH_returns','snp_returns','LINK_returns','XTZ_returns','BNB_returns','EOS_returns']

onchainFeats = ['DIFF', 'Hashrate_Price_Multiple', 'Mining Revenue', ]

bitfinexFeats =  ['longs_mean','shorts_mean', 'longs_max', 'shorts_max', 'longs_min', 'shorts_min','longs_close', 'shorts_close', 'longs_open', 'shorts_open']

bitmexFeats = [ 'fundingRateClose', 'fundingRateDailyClose', 'fundingRateOpen','fundingRateDailyOpen', 'fundingRateMean', 'fundingRateDailyMean','bitmex_OI_last', 'bitmex_OV_last','bitmex_OI_first', 'bitmex_OV_first']

glassnodeFeats =['exchange_outflow','exchange_inflow', 'stablecoin_supply_ratio', 'sopr','net_unrealized_profit_loss', 'mvrv']

technicalFeats = ['BTC_ma_21_rate','BTC_ma_21_std_center', 'BTC_ma_21_std', 'BTC_EMA_Cross','BTC_mayer_multiple']

feats = binanceFeats + rivalFeats + onchainFeats + bitfinexFeats + bitmexFeats + glassnodeFeats + technicalFeats

def selectTrainingFeatures(data2,fullpath2,printFeats=False, write=True):
	X = data2[feats]
	if write == True:
		X.to_csv(f"{fullpath2}/train.csv",index=False)
	return X

