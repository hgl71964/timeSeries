def sp500(fullpath,write=True):
  sp500 = pd.read_csv('https://stooq.com/q/d/l/?s=^spx&i=d')
  sp500['Date'] = pd.to_datetime(sp500['Date'])
  sp500['ohlc'] = sp500.apply(lambda row: (row.Open + row.Close +\
                                   row.High+row.Low)/4, axis = 1)
  sp500['returns'] = (sp500['close'] / sp500['close'].shift(1))
  sp500.iloc[0,-1] = 1
  sp500.columns = ["Date"] +["snp_"+ x for x in list(sp500.columns)[1:]]
  if write == True:
    sp500.to_csv(f"{fullpath}/sp500.csv",index=False)
  return sp500