import empyrical
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy
import statsmodels
from statsmodels.tsa.stattools import adfuller, coint
import plotly.express as px
import plotly.graph_objects as go

def plotCorrelation(WINDOW, tickers, temp):
  fig,ax = plt.subplots(figsize=(20,7.5),ncols=2)
  
  #correlation between returns
  sns.heatmap(temp[[x+  "_returns" for x in tickers]].iloc[-WINDOW:].corr(),annot=True,ax=ax[0])

  #correlation betwen closing prices
  sns.heatmap(temp[[x+"_close" for x in tickers]].iloc[-WINDOW:].corr(),annot=True,ax=ax[1])

def plotCointegration(WINDOW, tickers, temp):
  n = len(tickers)
  coints_prices = np.ones((n, n))
  coints_returns = np.ones((n, n))
  ticker_close = [x+  "_close" for x in tickers]
  ticker_returns = [x+  "_returns" for x in tickers]

  for i in range(n - 1):
    for j in range(i+1, n):
      tscore, p, thresholds = coint(temp[ticker_close[i]].iloc[-WINDOW:].values, temp[ticker_close[j]].iloc[-WINDOW:].values)
      coints_prices[i, j] = p

  for i in range(n - 1):
    for j in range(i+1, n):
      tscore, p, thresholds = coint(temp[ticker_returns[i]].iloc[-WINDOW:].fillna(1).values, temp[ticker_returns[j]].iloc[-WINDOW:].fillna(1).values)
      coints_returns[i, j] = p

  fig,ax = plt.subplots(figsize=(15,5),ncols=2)
  sns.heatmap(coints_returns,xticklabels=tickers,yticklabels=tickers,cmap="RdYlGn_r",annot=True,ax = ax[0])
  ax[0].set_title("Cointegration of Returns")
  sns.heatmap(coints_prices,xticklabels=tickers,yticklabels=tickers,cmap="RdYlGn_r",annot=True, ax = ax[1])
  ax[1].set_title("Cointegration of Prices")

def statistics(x):
  k, p = scipy.stats.normaltest(x, nan_policy='omit')
  mu = np.mean(x)
  std = np.std(x)
  adfuller_p = adfuller(x)[1]
  skew = f"Skew {scipy.stats.skew(x, nan_policy='omit')}"
  normal_fit = f"Goodness of Fit to Normal p-value: {p:.5f}"
  mean = f"Mean: {mu}"
  Adfuller = f"Adfuller p-value (test for stationarity): {adfuller_p}"
  SD = f"Standard Deviation: {std}"
  print(normal_fit+"\n"+Adfuller+"\n"+mean+"\n"+SD+"\n"+skew)
  return

def plotRatio(ticker1, ticker2, temp):
  temp[f"{ticker1}_{ticker2}_returns_ratio"] = (temp[f'{ticker1}_returns'] / temp[f'{ticker2}_returns'])
  temp[f"{ticker1}_{ticker2}_price_ratio"] = (temp[f'{ticker1}_close'] / temp[f'{ticker2}_close'])
  temp[f"{ticker1}_{ticker2}_price_ratio_diff"] = temp[f'{ticker1}_{ticker2}_price_ratio'].diff(1)
  print("Ratio of Returns")
  statistics(temp[f'{ticker1}_{ticker2}_returns_ratio'].iloc[1:])
  print("\nLog Ratio of Returns")
  statistics(np.log(temp[f'{ticker1}_{ticker2}_returns_ratio'].iloc[1:]))
  print("\nPrice Ratio")
  statistics(temp[f'{ticker1}_{ticker2}_price_ratio'].iloc[1:])
  print("\nPrice Ratio Difference")
  statistics(temp[f'{ticker1}_{ticker2}_price_ratio_diff'].iloc[1:])
  
  fig2, ax = plt.subplots(figsize=(20,10),ncols=3,nrows=2)
  sns.distplot(temp[f'{ticker1}_{ticker2}_returns_ratio'],kde=True,bins=30,ax=ax[0,0])
  ax[0,0].set_title("Distribution of the ratio of returns")
  sns.distplot(temp[f'{ticker1}_{ticker2}_price_ratio'],kde=True,bins=30,ax=ax[0,1])
  ax[0,1].set_title("Distribution of the ratio of prices")
  sns.distplot(temp[f'{ticker1}_{ticker2}_price_ratio_diff'],kde=True,bins=30,ax=ax[0,2])
  ax[0,2].set_title("Distribution of the price ratio diffs")
  sns.scatterplot(x=temp[f"{ticker1}_returns"],y=temp[f"{ticker2}_returns"],ax=ax[1,0])
  ax[1,0].set_title("Scatter of returns ")
  sns.scatterplot(x=temp[f"{ticker1}_close"],y=temp[f"{ticker2}_close"],ax=ax[1,1])
  ax[1,1].set_title("Scatter of prices")

  fig = px.line(x=temp['date'],y=temp[f'{ticker1}_{ticker2}_returns_ratio'], title=f"{ticker1}_{ticker2}_returns_ratio")
  fig.show()
  fig3 = px.line(x=temp['date'],y=temp[f'{ticker1}_{ticker2}_price_ratio'], title=f"{ticker1}_{ticker2}_price_ratio")
  fig3.show()
  fig4 = px.line(x=temp['date'],y=temp[f"{ticker1}_{ticker2}_price_ratio_diff"], title=f"{ticker1}_{ticker2}_price_ratio_diff")
  fig4.show()
  # fig2 = px.line(x=temp['date'],y=temp[f"{'{ticker1}_{ticker2}_returns_ratio'}"], title=f"{ticker1} to {ticker2} {kind} ratio")
  return  


#backtest helper function, returns a dataframe of trades
def backtest(temp,ticker1, ticker2, sellPValue=0.55, buyPValue=0.45, starting_cap=10000, transactionCostsPct=0, indicator="returns_ratio"):
  data = temp[[f'{ticker1}_open',f'{ticker2}_open',
               f'{ticker1}_high',f'{ticker2}_high',
               f'{ticker1}_low',f'{ticker2}_low',
               f'{ticker1}_close',f'{ticker2}_close',
               f'{ticker1}_returns',f'{ticker2}_returns',
               f'{ticker1}_{ticker2}_returns_ratio',f'{ticker1}_{ticker2}_price_ratio_diff','date']].copy()
  data[f'{ticker1}_{ticker2}_returns_ratio_last'] = data[f'{ticker1}_{ticker2}_returns_ratio'].shift(1)
  data[f'{ticker1}_{ticker2}_price_ratio_diff_last'] = data[f'{ticker1}_{ticker2}_price_ratio_diff'].shift(1)



  #units of cash, units of ticker1, units of ticker2, traded, time , equity, returns
  data['cash'] = 0
  data['value'] = 0
  data['traded'] = 0
  data[f'{ticker1}_unit'] = 0
  data[f'{ticker2}_unit'] = 0


  # #initialise initial positions
  data.loc[1,["cash","value"]] = starting_cap
  if indicator == 'price_ratio_diff':
    mu = data[f'{ticker1}_{ticker2}_price_ratio_diff'].mean()
    std = data[f'{ticker1}_{ticker2}_price_ratio_diff'].std()
  else:
    mu = data[f'{ticker1}_{ticker2}_returns_ratio'].mean()
    std = data[f'{ticker1}_{ticker2}_returns_ratio'].std()

  for i in range(2,data.shape[0]):
    ticker1_price = data.iloc[i][f"{ticker1}_open"]
    ticker2_price = data.iloc[i][f'{ticker2}_open']
    data.loc[i,["cash",f'{ticker1}_unit', f'{ticker2}_unit']] = data.loc[i-1,["cash",f'{ticker1}_unit', f'{ticker2}_unit']]
  
    #kelly criterion position sizing
    if indicator == 'price_ratio_diff':
      signal = data.loc[i,f'{ticker1}_{ticker2}_price_ratio_diff_last']
    else:
      signal = data.loc[i,f'{ticker1}_{ticker2}_returns_ratio_last']
    p  = scipy.stats.norm.cdf(signal,loc=mu,scale=std)
    
    #cost to long 1 unit of the ratio
    cost = (ticker1_price - ticker2_price)
    #if the ratio is highly above mean; set your proportion of equity to be p
    if p > sellPValue: #p > 1
      data.loc[i,'traded'] = -1                                  
      data.loc[i,f'{ticker1}_unit'] = p * data.loc[i-1, 'value'] / cost
      data.loc[i,f'{ticker2}_unit'] = -p  * data.loc[i-1, 'value'] / cost
      #buy ticker1
      data.loc[i,"cash"] -= (data.loc[i,f'{ticker1}_unit'] - data.loc[i-1,f'{ticker1}_unit']) * cost
  #   #if the ratio is significantly below mean;
    elif p < buyPValue:
      data.loc[i,'traded'] = 1
      data.loc[i,f'{ticker1}_unit'] = p * data.loc[i-1, 'value'] / cost
      data.loc[i,f'{ticker2}_unit'] = -p * data.loc[i-1, 'value']/ cost
      #buy ticker1
      data.loc[i,"cash"] -=  (data.loc[i,f'{ticker1}_unit'] - data.loc[i-1,f'{ticker1}_unit']) * cost
      
  #   #exit positions if they are trading in a normal range
    else:
      #0 positions
      data.loc[i,f'{ticker1}_unit'] = 0
      data.loc[i,f'{ticker2}_unit'] = 0
      data.loc[i,"cash"] += data.loc[i-1,f'{ticker1}_unit'] * cost

    value = data.loc[i,f'{ticker1}_close'] * data.loc[i,f'{ticker1}_unit'] + data.loc[i,f'{ticker2}_close'] * data.loc[i,f'{ticker2}_unit'] + data.iloc[i]['cash']
    data.loc[i,"value"] = value
  
  data['returns'] = data['value']  / data['value'].shift(1)
  # backtest = pd.merge(positions.iloc[1:], data,on='date')
  return data

def diagnosis(backtestResults, ticker1, ticker2):
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=backtestResults.loc[2:,'date'], y=np.exp(np.log(backtestResults.loc[2:,'returns']).cumsum()),
                    mode='markers+lines',
                    marker_color=backtestResults.loc[2:,'traded'],
                    name='Strategy Returns (Cumulative)'))
  fig.add_trace(go.Scatter(x=backtestResults.loc[2:,'date'], y=np.exp(np.log(backtestResults.loc[2:,f'{ticker1}_returns']).cumsum()),
                    mode='lines',
                    name=f'{ticker1} Returns (Cumulative)'))
  fig.add_trace(go.Scatter(x=backtestResults.loc[2:,'date'], y=np.exp(np.log(backtestResults.loc[2:,f'{ticker2}_returns']).cumsum()),
                    mode='lines',
                    name=f'{ticker2} Returns (Cumulative)'))
  fig.show()

  fig2 = go.Figure()
  fig2.add_trace(go.Scatter(x=backtestResults.loc[2:,'date'], y=backtestResults.loc[2:,'returns'],
                    mode='markers+lines',
                    marker_color=backtestResults.loc[2:,'traded'],
                    name='Strategy Returns'))
  fig2.add_trace(go.Scatter(x=backtestResults.loc[2:,'date'], y=backtestResults.loc[2:,f'{ticker1}_returns'],
                    mode='lines',
                    name=f'{ticker1} Returns'))
  fig2.add_trace(go.Scatter(x=backtestResults.loc[2:,'date'], y=backtestResults.loc[2:,f'{ticker2}_returns'],
                    mode='lines',
                    name=f'{ticker2} Returns'))
  fig2.show()

  
  sharpe = empyrical.sharpe_ratio(backtestResults.loc[2:,'returns']-1,risk_free=0)
  md = empyrical.max_drawdown(backtestResults.loc[2:,'returns']-1)
  sortino = empyrical.sortino_ratio(backtestResults.loc[2:,'returns']-1,required_return=1.07**(1/365)-1)
  print(f"Sharpe: {sharpe}\n Max Drawdown: {md}\n Sortino: {sortino}")
  return