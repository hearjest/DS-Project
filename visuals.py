import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import os
from IPython.display import display
file_path = "test4.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} not found. Make sure your file is uploaded.")

# Load CSV
df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

# Handle tz-aware datetimes safely
try:
    idx = pd.to_datetime(df.index)
    # If tz-aware, convert to UTC then drop tz; otherwise keep
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    df.index = idx
except Exception:
    # Fallback: coerce and remove tz
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)


required_cols = ['Open','High','Low','Close','Volume','Dividends','Stock Splits']
for c in required_cols:
    if c not in df.columns:
        raise KeyError(f"Required column {c} not found in CSV columns: {list(df.columns)}")


if 'Adj Close' not in df.columns:
    df['Adj Close'] = df['Close']


df = df.sort_index()


missing_counts = df.isna().sum()


df['return'] = df['Adj Close'].pct_change()

# Choose threshold for "Same" class:
delta = 0.005  # 0.5%
def label_return(r, delta=delta):
    if pd.isna(r):
        return np.nan
    if r > delta:
        return "Up"
    elif r < -delta:
        return "Down"
    else:
        return "Same"
df['label'] = df['return'].apply(lambda x: label_return(x))

# Drop first row (NaN return)
df = df.dropna(subset=['return'])

class_counts = df['label'].value_counts()

df['ret_lag1'] = df['return'].shift(1)
df['ret_lag2'] = df['return'].shift(2)
df['ma5'] = df['Adj Close'].rolling(window=5).mean().pct_change() 
df['ma10'] = df['Adj Close'].rolling(window=10).mean().pct_change()
df['vol30'] = df['return'].rolling(window=30).std()  
df['vol10'] = df['return'].rolling(window=10).std()
df['vol_ratio'] = df['vol10'] / (df['vol30'] + 1e-12)  
df['vol_lag1'] = df['vol30'].shift(1)
df['vol_lag2'] = df['vol30'].shift(2)
df['volatility_rolling'] = df['return'].rolling(window=30).std() * np.sqrt(252)  
df['volume_lag1'] = df['Volume'].shift(1)
df['vol_change'] = df['Volume'] / (df['volume_lag1'] + 1e-12) - 1

# Drop rows with NaN in features

# Plot 1: Adjusted Close price time series (full)
plt.figure(figsize=(10,4))
plt.plot(df['Adj Close'])
plt.title("AAPL Adjusted Close Price (full series)")
plt.xlabel("Date")
plt.ylabel("Adj Close")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Histogram of daily returns (zoomed)
plt.figure(figsize=(8,4))
plt.hist(df['return'].dropna().values, bins=80)
plt.title("Histogram of daily returns (Adj Close)")
plt.xlabel("Daily return")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Plot 3: Rolling volatility (30-day)
plt.figure(figsize=(10,4))
plt.plot(df['volatility_rolling'])
plt.title("Rolling 30-day volatility (annualized approx)")
plt.xlabel("Date")
plt.ylabel("Volatility (approx annualized)")
plt.tight_layout()
plt.show()

# Plot 4: Class distribution bar chart (overall)
plt.figure(figsize=(8,4))
plt.bar(class_counts.index.astype(str), class_counts.values)
plt.title("Overall class distribution (Up / Down / Same)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
