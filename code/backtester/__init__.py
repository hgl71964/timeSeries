import numpy as np
import pandas as pd

def backtest(X_test,models, data2, transaction_costs):
  test_df = pd.DataFrame(models['cbc'].predict_proba(X_test)[:,1])
  test_df['xgbc'] = pd.DataFrame(models['xgbc'].predict_proba(X_test)[:,1])
  test_df.columns=['cbc','xgbc']
  signals =  models['ensemble'].predict_proba(test_df)[:,1]

  data = X_test.copy()
  data['close'] = data2.loc[X_test.index[0]:,'BTC_close']
  data = data.reset_index(drop=True)


  starting_cap = 10000

  #units of cash, units of btc
  positions = np.array([[0.0 for i in range(4)] for j in range(len(signals)+1)])
  positions[0,0] = starting_cap
  positions[0,3] = starting_cap
  positions = pd.DataFrame(positions)
  positions.columns=["cash", "btc","traded","value"]


  for i in range(len(signals)):
    entry_price = data.iloc[i]['close']
    positions.iloc[i+1,0:2] = positions.iloc[i,0:2]
    p = abs(signals[i] - 0.5)
    if signals[i] > 0.5:
      entry_capital = positions.iloc[i,0] * p 
      if entry_capital > 0:
        positions.iloc[i+1,0] -= float(positions.iloc[i,0]) * p 
        positions.iloc[i+1,1] += float(positions.iloc[i,0]) * p  / (float(entry_price) * (1 + transaction_costs))
        positions.iloc[i+1,2] = 1
    elif signals[i] < 0.5:
        if positions.iloc[i, 1] > 0:
          positions.iloc[i+1,0] += positions.iloc[i,1] * entry_price * (1 - transaction_costs)
          positions.iloc[i+1,1] = 0
          positions.iloc[i+1,2] = -1
    value = entry_price * positions.iloc[i+1,1] + positions.iloc[i+1,0]
    positions.iloc[i+1,3] = value
  return positions

