import numpy as np
import pandas as pd


def backtest(X_test,models, data2, transaction_costs, ls = False):
  test_df = pd.DataFrame(models['cbc'].predict_proba(X_test)[:,1])
  test_df['xgbc'] = pd.DataFrame(models['xgbc'].predict_proba(X_test)[:,1])
  test_df.columns=['cbc','xgbc']
  signals =  models['ensemble'].predict_proba(test_df)[:,1]

  data = X_test.copy()
  data['close'] = data2.loc[X_test.index[0]:,'BTC_close']
  data = data.reset_index(drop=True)


  starting_cap = 10000

  #units of cash, units of btc
  positions = np.array([[0.0 for i in range(6)] for j in range(len(signals)+1)])
  positions[0,0] = starting_cap #set startin cash to starting_cap
  positions[0,3] = starting_cap #set starting value to starting_value
  positions = pd.DataFrame(positions)
  positions.columns=["cash", "btc","traded","value", 'btc_close','date']
  positions.loc[1:,'btc_close'] = data['close']
  positions['date'] = data2.loc[X_test.index[0]:,'date'].reset_index(drop=True)

  for i in range(len(signals)):
    entry_price = data.iloc[i]['close']
    positions.iloc[i+1,0:2] = positions.iloc[i,0:2]
    p = signals[i]
    if ls == False:
        targetValue = positions.iloc[i,3] * p
        if targetValue > positions.iloc[i+1,1] * entry_price:
          positions.iloc[i+1,1] = targetValue / (entry_price * (1 + transaction_costs))
          positions.iloc[i+1,0] -= (positions.iloc[i+1,1]- positions.iloc[i,1]) * entry_price * (1 + transaction_costs)
          positions.iloc[i+1,2] = 1
        else:
          positions.iloc[i+1,1] = targetValue / (entry_price - (1 + transaction_costs))
          positions.iloc[i+1,0] -= (positions.iloc[i+1,1] - positions.iloc[i,1]) * entry_price * (1 - transaction_costs)
          positions.iloc[i+1,2] = -1
    if ls == True:
        targetValue = positions.iloc[i,3] * (2 * p - 1)
        if signals[i] > 0.5:
          positions.iloc[i+1,1] = targetValue / (entry_price * (1 + transaction_costs))
          positions.iloc[i+1,0] -= (positions.iloc[i+1,1]- positions.iloc[i,1]) * entry_price * (1 + transaction_costs)
          positions.iloc[i+1,2] = 1
        else:
          positions.iloc[i+1,1] = targetValue / (entry_price - (1 + transaction_costs))
          positions.iloc[i+1,0] -= (positions.iloc[i+1,1] - positions.iloc[i,1]) * entry_price * (1 - transaction_costs)
          positions.iloc[i+1,2] = -1
    value = entry_price * positions.iloc[i+1,1] + positions.iloc[i+1,0]
    positions.iloc[i+1,3] = value


  positions['returns'] = positions['value'] / positions['value'].shift(1)
  dates = data2.loc[X_test.index[0]:,'date']
  y1 = np.exp(np.log(positions['returns']).cumsum()).iloc[1:]
  y2 = np.exp(np.log(data2.loc[X_test.index[0]:,'BTC_returns'].shift(1)).cumsum())
  fig = go.Figure(go.Scatter(x=dates,y=y1.iloc[1:],marker_color=positions['traded'], mode='markers+lines', name='Strategy returns (%)'))
  fig.add_trace(go.Scatter(x=dates, y=y2.iloc[1:], name='Buy and Hold Returns (%)'))
  layout = go.Layout(yaxis=dict(tickformat=".2%"))

  sharpe = empyrical.sharpe_ratio(positions.loc[1:,'returns']-1,risk_free=0)
  md = empyrical.max_drawdown(positions.loc[1:,'returns']-1)
  sortino = empyrical.sortino_ratio(positions.loc[1:,'returns']-1,required_return=1.07**(1/365)-1)
  print(f"Sharpe: {sharpe}\n Max Drawdown: {md}\n Sortino: {sortino}")
  
  return positions.iloc[1:], fig

