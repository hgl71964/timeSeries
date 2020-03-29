import empyrical
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def backtest(X_test,models, data2, transaction_costs,starting_cap = 10000,  confidence=0, longLimit=1, shortLimit=-1):
  test_df = pd.DataFrame(models['cbc'].predict_proba(X_test)[:,1])
  test_df['xgbc'] = pd.DataFrame(models['xgbc'].predict_proba(X_test)[:,1])
  test_df.columns=['cbc','xgbc']
  signals = models['ensemble'].predict_proba(test_df)[:,1]
  #units of cash, units of btc
  positions = np.array([[0.0 for i in range(7)] for j in range(len(signals)+1)])
  positions[0,0] = starting_cap #set startin cash to starting_cap
  positions[0,3] = starting_cap #set starting value to starting_value
  positions = pd.DataFrame(positions)
  positions.columns=["cash", "btc_units","traded","value", 'btc_open','date', 'signal']
  positions.loc[1:,'btc_open'] = data2.loc[X_test.index[0]:,'BTC_open'].reset_index(drop=True)
  positions.loc[1:,'date'] = data2.loc[X_test.index[0]:,'date'].values[:-1]

  positions.loc[1:,'signal'] = signals

  for i in range(len(signals)):
    entry_price = positions.iloc[i+1, 4]
    positions.iloc[i+1,0:2] = positions.iloc[i,0:2]
    p = positions.iloc[i+1,6]
  
    if signals[i] >= (0.5 + (float(confidence) / 2)): #long signals
      targetValue = positions.iloc[i,3] * min((2 * p - 1), longLimit)
      positions.iloc[i+1,1] = targetValue / (entry_price * (1 + transaction_costs))
      positions.iloc[i+1,0] -= (positions.iloc[i+1,1]- positions.iloc[i,1]) * entry_price * (1 + transaction_costs)
      positions.iloc[i+1,2] = 1
    elif signals[i] <= (0.5 - (float(confidence) / 2)): # short signal
      targetValue = positions.iloc[i,3] * max((2 * p - 1), shortLimit)
      positions.iloc[i+1,1] = targetValue / (entry_price - (1 - transaction_costs))
      positions.iloc[i+1,0] -= (positions.iloc[i+1,1] - positions.iloc[i,1]) * entry_price * (1 - transaction_costs)
      positions.iloc[i+1,2] = -1
    else: #go to cash if not confident enough
      btcUnits = positions.iloc[i+1,1]
      positions.iloc[i+1,1] = 0
      if btcUnits < 0:
        positions.iloc[i+1, 0] += btcUnits * entry_price * (1 + transaction_costs)
      else:
        positions.iloc[i+1, 0]  += btcUnits * entry_price * (1 - transaction_costs)

    value = entry_price * positions.iloc[i+1,1] + positions.iloc[i+1,0]
    positions.iloc[i+1,3] = value

  positions['returns'] = positions['value'] / positions['value'].shift(1)
  positions = positions.iloc[1:]
  y1 = np.exp(np.log(positions.iloc[1:]['returns']).cumsum())
  y2 = np.exp(np.log(data2.loc[X_test.index[0]+1:,'BTC_returns']).cumsum())
  fig = go.Figure(go.Scatter(x=positions.loc[1:,'date'],y=y1,marker_color=positions['traded'], mode='markers+lines', name='Strategy returns (%)'))
  fig.add_trace(go.Scatter(x=positions.loc[1:,'date'], y=y2, name='Buy and Hold Returns (%)'))
  layout = go.Layout(yaxis=dict(tickformat=".2%"))

  sharpe = empyrical.sharpe_ratio(positions.loc[1:,'returns']-1,risk_free=0)
  md = empyrical.max_drawdown(positions.loc[1:,'returns']-1)
  sortino = empyrical.sortino_ratio(positions.loc[1:,'returns']-1,required_return=1.07**(1/365)-1)
  expectedGain = positions.loc[(positions['returns'] > 1,'returns')].mean()
  expectedLoss = positions.loc[(positions['returns'] <= 1,'returns')].mean()
  winRate = (positions['returns'] > 1).mean()
  print(f"Sharpe: {sharpe}")
  print(f"Max Drawdown: {md}")
  print(f"Sortino: {sortino}")
  print(f"Win Rate: WinRate:{100*winRate:.5f}%")
  print(f"expected gain on winning trade: {100*(expectedGain-1):.5f} %")
  print(f"expected loss on losing trade: {100*(expectedLoss-1):.5f} %")
  
  return positions, fig


def backtester2(predictions, original_df, X_test, starting_cap = 10000,  longLimit=1, shortLimit=-1):
  signals = predictions #predicted returns
  #units of cash, units of btc
  #initialise dataframe
  positions = np.array([[0.0 for i in range(7)] for j in range(len(signals)+1)])
  positions[0,0] = starting_cap #set startin cash to starting_cap
  positions[0,3] = starting_cap #set starting value to starting_value
  positions = pd.DataFrame(positions)
  positions.columns=["cash", "btc_units","traded","value", 'btc_open','date', 'signal']
  #start counting prices, date from the start of testing period
  positions.loc[1:,'btc_open'] = original_df.loc[original_df.shape[0]-X_test.shape[0]-1:,'BTC_open'].reset_index(drop=True)
  positions.loc[1:,'date'] = pd.to_datetime(original_df.loc[original_df.shape[0]-X_test.shape[0]-1:,'date'].values[:-1])
  positions.loc[1:,'signal'] = signals
  
  #loop through and backtest
  for i in range(len(signals)):
    entry_price = positions.iloc[i+1, 4] #purchase at open price
    positions.iloc[i+1,0:2] = positions.iloc[i,0:2] #carry forward cash and bitcoin from prior position
    p = 2 * (positions.iloc[i+1,6] > 1) - 1 # 1 if predicted above, -1 if predicted below
    if p > 0 :
      targetValue = positions.iloc[i,3] * min(p, longLimit) #e.g. min(1, 0.5)
    else:
      targetValue = positions.iloc[i,3] * max(p, shortLimit) #e.g. max(-1, -0.5)
    positions.iloc[i+1,1] = targetValue / entry_price #new Units
    positions.iloc[i+1,0] -= (positions.iloc[i+1,1]- positions.iloc[i,1]) * entry_price #change in cash
    if targetValue > positions.iloc[i+1,1] * entry_price: # if not enough in position, expand it
          positions.iloc[i+1,2] = 1 #went long
    else: # else reduce position
          positions.iloc[i+1,2] = -1 #went short
      
    positions.iloc[i+1,3] = entry_price * positions.iloc[i+1,1] + positions.iloc[i+1,0] #value = cash + value of BTC

  positions['returns'] = positions['value'] / positions['value'].shift(1) #calculate returns at end
  positions = positions.iloc[1:]
  y1 = np.exp(np.log(positions.iloc[1:]['returns']).cumsum()) #cumulative returns
  y2 = np.exp(np.log(original_df.loc[original_df.shape[0]-X_test.shape[0]:,'BTC_returns']).cumsum()) #cumulative strategy returns
  #plot
  fig = go.Figure(go.Scatter(x=positions.loc[1:,'date'],y=y1,marker_color=positions['traded'], mode='markers+lines', name='Strategy returns (%)'))
  fig.add_trace(go.Scatter(x=positions.loc[1:,'date'], y=y2, name='Buy and Hold Returns (%)'))
  fig.update_layout(title='Backtest Performance')
  layout = go.Layout(yaxis=dict(tickformat=".2%"))

  actual_rets = (original_df['BTC_close'].shift(-1) / original_df['BTC_close']).loc[original_df.shape[0]-X_test.shape[0]-1:]
  fig2 = go.Figure(go.Scatter(x=positions.loc[1:,'date'],y=actual_rets.iloc[:-1], name='Actual Returns'))
  fig2.add_trace(go.Scatter(x=positions.loc[2:,'date'],y=signals,name='Predicted Returns'))
  fig2.update_layout(title='Predicted vs Actual')

  #summary statistics
  sharpe = empyrical.sharpe_ratio(positions.loc[1:,'returns']-1,risk_free=0) # sharpe ratio
  md = empyrical.max_drawdown(positions.loc[1:,'returns']-1) #maximum drawdown
  sortino = empyrical.sortino_ratio(positions.loc[1:,'returns']-1,required_return=1.07**(1/365)-1) #sortino ratio
  expectedGain = positions.loc[(positions['returns'] > 1,'returns')].mean()
  expectedLoss = positions.loc[(positions['returns'] <= 1,'returns')].mean()
  winRate = (positions['returns'] > 1).mean()
  print("BACKTEST RESULTS")
  print("--------------------------")
  print(f"Sharpe: {sharpe}\nSortino: {sortino}")
  print(f"Max Drawdown: {100*md:.5f}%")
  print(f"Win Rate: {100*winRate:.5f}%")
  print(f"expected gain on winning trade: {100*(expectedGain-1):.5f} %")
  print(f"expected loss on losing trade: {100*(expectedLoss-1):.5f} %")
  
  return positions, fig, fig2
