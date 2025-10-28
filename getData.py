import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
dat = yf.Ticker("AAPL")
#print(dat.history(period='5d',interval="1d",prepost=False,auto_adjust=True,actions=True))
#history=dat.history(period='30y',interval="1d",prepost=False,auto_adjust=True,actions=True)
#file=pd.DataFrame(history)
options=dat.option_chain(dat.options[0]).calls
optionsFile=pd.DataFrame(options)
optionsFile.to_csv("options.csv")
print(options)
#file=pd.DataFrame(history)
#print(file['Datetime'])
#file.to_csv("test4.csv")
#plt.plot(file.index,file['Open'])
#plt.show()
