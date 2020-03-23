import empyrical
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def backtest(X_test,models, data2, transaction_costs,starting_cap = 10000,  confidence=0, longLimit=1, shortLimit=-1, ls = False):
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
  positions.loc[1:,'date'] = data2.loc[X_test.index[0]:,'date'].reset_index(drop=True)

  positions.loc[1:,'signal'] = signals

  for i in range(len(signals)):
    entry_price = positions.iloc[i+1, 4]
    positions.iloc[i+1,0:2] = positions.iloc[i,0:2]
    p = positions.iloc[i+1,6]
    if ls == False:
        targetValue = positions.iloc[i,3] * p
        if abs(signals[i] - 0.5) * 2 >  confidence:
          if targetValue > positions.iloc[i+1,1] * entry_price:
            positions.iloc[i+1,1] = targetValue / (entry_price * (1 + transaction_costs))
            positions.iloc[i+1,0] -= (positions.iloc[i+1,1]- positions.iloc[i,1]) * entry_price * (1 + transaction_costs)
            positions.iloc[i+1,2] = 1
          else:
            positions.iloc[i+1,1] = targetValue / (entry_price - (1 - transaction_costs))
            positions.iloc[i+1,0] -= (positions.iloc[i+1,1] - positions.iloc[i,1]) * entry_price * (1 - transaction_costs)
            positions.iloc[i+1,2] = -1
    if ls == True:
        if signals[i] > 0.5 + float(confidence) / 2:
          targetValue = positions.iloc[i,3] * min((2 * p - 1), longLimit)
          positions.iloc[i+1,1] = targetValue / (entry_price * (1 + transaction_costs))
          positions.iloc[i+1,0] -= (positions.iloc[i+1,1]- positions.iloc[i,1]) * entry_price * (1 + transaction_costs)
          positions.iloc[i+1,2] = 1
        elif signals[i] <= 0.5 - float(confidence) / 2:
          targetValue = positions.iloc[i,3] * max((2 * p - 1), shortLimit)
          positions.iloc[i+1,1] = targetValue / (entry_price - (1 - transaction_costs))
          positions.iloc[i+1,0] -= (positions.iloc[i+1,1] - positions.iloc[i,1]) * entry_price * (1 - transaction_costs)
          positions.iloc[i+1,2] = -1
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
  print(f"Sharpe: {sharpe}\nMax Drawdown: {md}\nSortino: {sortino}")
  
  return positions, fig

