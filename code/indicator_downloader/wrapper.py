def getLatestData():
	data = downloadWrapper(tickers, API_SECRET, API_KEY, FREQ, fullpath, write=True)