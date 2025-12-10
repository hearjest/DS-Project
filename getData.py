import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

dat = yf.Ticker("AAPL")
<<<<<<< HEAD
 
=======

>>>>>>> 5da5bcf (update)
# Get historical price data
print("Downloading historical price data...")
history = dat.history(period='40y', interval="1d", prepost=False, auto_adjust=True, actions=True)
file = pd.DataFrame(history)
file.to_csv("test5.csv")
print(f"Historical data saved to test5.csv ({len(file)} rows)")
print("\nNote: Options data, NASDAQ, DOW, and S&P 500 data downloads have been removed.")
