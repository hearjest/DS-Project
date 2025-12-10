"""

alpha vantage api X05FO1W6ZVSZI34R
Enhanced Feature Extraction with Options Data and Market Context

This module extends the base feature extraction by adding:
1. Options data features (from yfinance)
2. Market context features (SPY, VIX)
3. Technical indicators

OPTIONS DATA EXPLANATION:
- Implied Volatility (IV): Market's expectation of future volatility
- Put/Call Ratio: Ratio of put to call options (high = bearish sentiment)
- Options Volume: Trading activity in options (high = increased interest)
- Open Interest: Number of outstanding contracts
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Configuration
RAW_CSV = "test5.csv"  # Default: matches getData.py output; script will try test4.csv if test5.csv not found
PROCESSED_CSV = "./processedtest4_enhanced2.csv"
CLEANING_LOG = "cleaning_log_enhanced.csv"
STOCK_TICKER = "AAPL"
DELTA = 0.005  # Threshold for "Same" class (0.5% - increased from 0.3% to improve class balance)
EXTREME_THRESHOLD = 0.10

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

def load_raw(path):
    """Load raw stock price data"""
    # Try the specified path first, then try alternative names
    if not os.path.exists(path):
        # Try alternative file names
        alt_paths = ["test5.csv", "test4.csv", "DS-Project/test5.csv", "DS-Project/test4.csv"]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"[INFO] Using alternative path: {alt_path}")
                path = alt_path
                break
        else:
            raise FileNotFoundError(f"Raw CSV not found: {path}. Also tried: {alt_paths}. Run getData.py first.")
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    try:
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    except Exception:
        df.index = pd.to_datetime(df.index).tz_localize(None, ambiguous='infer', nonexistent='shift_forward')
    df = df.sort_index()
    return df

def price_consistency_check(df):
    """Check for logical price inconsistencies"""
    cond = (
        (df['Open'] > df['High']) |
        (df['Close'] > df['High']) |
        (df['Low'] > df['Open']) |
        (df['Low'] > df['Close'])
    )
    violations = df[cond]
    if len(violations) > 0:
        os.makedirs("data", exist_ok=True)
        violations.reset_index().to_csv("data/price_consistency_violations.csv", index=False)
        df = df.drop(index=violations.index)
        print(f"[CLEAN] Price consistency: dropped {len(violations)} rows")
    else:
        print("[CLEAN] Price consistency: no violations found.")
    return df

def get_options_features(ticker_symbol, date):
    """
    Extract options features for a given date.
    
    Returns a dictionary with:
    - iv_mean: Average implied volatility across all options
    - iv_std: Standard deviation of IV (shows dispersion)
    - put_call_volume_ratio: Put volume / Call volume (sentiment indicator)
    - put_call_oi_ratio: Put open interest / Call open interest
    - total_options_volume: Total options trading volume
    - atm_iv: At-the-money implied volatility (most relevant)
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Get available expiration dates
        exp_dates = ticker.options
        if len(exp_dates) == 0:
            return None
        
        # Use nearest expiration date (most liquid)
        nearest_exp = exp_dates[0]
        
        # Get options chain
        try:
            opt_chain = ticker.option_chain(nearest_exp)
            calls = opt_chain.calls
            puts = opt_chain.puts
        except Exception as e:
            print(f"[OPTIONS] Error fetching options for {date}: {e}")
            return None
        
        if len(calls) == 0 and len(puts) == 0:
            return None
        
        # Calculate features
        features = {}
        
        # Implied Volatility features
        all_iv = []
        if len(calls) > 0:
            all_iv.extend(calls['impliedVolatility'].dropna().values)
        if len(puts) > 0:
            all_iv.extend(puts['impliedVolatility'].dropna().values)
        
        if len(all_iv) > 0:
            features['iv_mean'] = np.mean(all_iv)
            features['iv_std'] = np.std(all_iv)
        else:
            features['iv_mean'] = np.nan
            features['iv_std'] = np.nan
        
        # At-the-money IV (closest to current price)
        # We'll approximate this by using the mean of near-ATM options
        if len(calls) > 0:
            # Get current price from the most recent data point
            # For now, use mean of calls with strike near current price
            features['atm_iv'] = calls['impliedVolatility'].dropna().mean() if len(calls) > 0 else np.nan
        else:
            features['atm_iv'] = np.nan
        
        # Put/Call Ratios (sentiment indicators)
        # High put/call ratio = bearish sentiment
        if len(calls) > 0 and len(puts) > 0:
            call_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
            put_volume = puts['volume'].sum() if 'volume' in puts.columns else 0
            call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
            put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0
            
            features['put_call_volume_ratio'] = put_volume / (call_volume + 1e-12)
            features['put_call_oi_ratio'] = put_oi / (call_oi + 1e-12)
            features['total_options_volume'] = call_volume + put_volume
        else:
            features['put_call_volume_ratio'] = np.nan
            features['put_call_oi_ratio'] = np.nan
            features['total_options_volume'] = np.nan
        
        return features
    except Exception as e:
        print(f"[OPTIONS] Error processing options for {date}: {e}")
        return None

def get_market_context_features(start_date, end_date):
    """
    Fetch market context features:
    - SPY returns (market performance)
    - VIX (volatility index - fear gauge)
    - Market correlation
    """
    features_dict = {}
    
    try:
        # Get SPY data (S&P 500 ETF - market proxy)
        spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
        if len(spy) > 0:
            # Handle MultiIndex columns (yfinance sometimes returns this)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.droplevel(1)
            
            # Try 'Adj Close' first, fall back to 'Close'
            price_col = 'Adj Close' if 'Adj Close' in spy.columns else 'Close'
            if price_col in spy.columns:
                spy['return'] = spy[price_col].pct_change()
                features_dict['spy_return'] = spy['return']
                features_dict['spy_volatility'] = spy['return'].rolling(30).std()
            else:
                raise ValueError("No price column found in SPY data")
        else:
            features_dict['spy_return'] = pd.Series(dtype=float)
            features_dict['spy_volatility'] = pd.Series(dtype=float)
    except Exception as e:
        print(f"[MARKET] Error fetching SPY data: {e}")
        features_dict['spy_return'] = pd.Series(dtype=float)
        features_dict['spy_volatility'] = pd.Series(dtype=float)
    
    try:
        # Get VIX data (Volatility Index - fear gauge)
        # High VIX = high fear/uncertainty
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        if len(vix) > 0:
            # Handle MultiIndex columns
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.droplevel(1)
            
            # Try 'Adj Close' first, fall back to 'Close'
            price_col = 'Adj Close' if 'Adj Close' in vix.columns else 'Close'
            if price_col in vix.columns:
                features_dict['vix'] = vix[price_col]
                features_dict['vix_change'] = vix[price_col].pct_change()
            else:
                raise ValueError("No price column found in VIX data")
        else:
            features_dict['vix'] = pd.Series(dtype=float)
            features_dict['vix_change'] = pd.Series(dtype=float)
    except Exception as e:
        print(f"[MARKET] Error fetching VIX data: {e}")
        features_dict['vix'] = pd.Series(dtype=float)
        features_dict['vix_change'] = pd.Series(dtype=float)
    
    return features_dict

def add_technical_indicators(df):
    """
    Add technical indicators:
    - RSI: Relative Strength Index (momentum, 0-100, >70 overbought, <30 oversold)
    - MACD: Moving Average Convergence Divergence (trend)
    - Bollinger Bands: Volatility bands
    """
    # RSI (Relative Strength Index)
    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-12)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['Adj Close'].rolling(window=20).mean()
    bb_std = df['Adj Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['Adj Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-12)
    
    return df

def label_return(r, delta=DELTA):
    """
    BINARY classification: Up or Down only (no Same class)
    Returns "Up" if return > delta, "Down" otherwise
    """
    if pd.isna(r):
        return np.nan
    if r > delta:
        return "Up"
    else:
        return "Down"  # Includes neutral/small moves as "Down"

def create_enhanced_features(df, use_options=True, use_market_context=True):
    """
    Create enhanced features with options data and market context.
    
    Note: Options data from yfinance is limited to recent dates.
    For historical data, you'd need a paid API.
    """
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']
    
    df['return'] = df['Adj Close'].pct_change()
    
    # Create label (shifted forward to predict tomorrow)
    df['return_next'] = df['return'].shift(-1)
    df['label'] = df['return_next'].apply(lambda x: label_return(x))
    
    n_before = len(df)
    
    # ===== IMPROVED BASE FEATURES (for next-day prediction) =====
    
    # Lagged returns (momentum indicators)
    df['ret_lag1'] = df['return'].shift(1)  # Yesterday's return
    df['ret_lag2'] = df['return'].shift(2)  # Day before yesterday
    df['ret_lag3'] = df['return'].shift(3)  # 3 days ago (additional momentum)
    
    # Moving averages (using yesterday's values to avoid look-ahead)
    ma5 = df['Adj Close'].rolling(window=5).mean()
    ma10 = df['Adj Close'].rolling(window=10).mean()
    ma20 = df['Adj Close'].rolling(window=20).mean()
    
    # Price position relative to MAs (known at end of day)
    df['price_vs_ma5'] = (df['Adj Close'] - ma5) / (ma5 + 1e-12)  # % above/below 5-day MA
    df['price_vs_ma10'] = (df['Adj Close'] - ma10) / (ma10 + 1e-12)  # % above/below 10-day MA
    df['price_vs_ma20'] = (df['Adj Close'] - ma20) / (ma20 + 1e-12)  # % above/below 20-day MA
    
    # MA slopes (rate of change in trend - using yesterday's values)
    df['ma5_slope'] = ma5.pct_change().shift(1)  # Yesterday's MA5 change
    df['ma10_slope'] = ma10.pct_change().shift(1)  # Yesterday's MA10 change
    
    # Volatility features (using yesterday's values to avoid look-ahead)
    vol30 = df['return'].rolling(window=30).std()
    vol10 = df['return'].rolling(window=10).std()
    df['vol30_lag1'] = vol30.shift(1)  # Yesterday's 30-day volatility
    df['vol10_lag1'] = vol10.shift(1)  # Yesterday's 10-day volatility
    df['vol_ratio'] = df['vol10_lag1'] / (df['vol30_lag1'] + 1e-12)  # Short-term vs long-term vol
    df['vol30_lag2'] = vol30.shift(2)  # 2 days ago volatility
    
    # Price action features (all known at end of day)
    df['high_low_range'] = (df['High'] - df['Low']) / (df['Adj Close'] + 1e-12)  # Daily range as % of price
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-12)  # Close position in daily range (0-1)
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-12)  # Gap from previous close
    
    # Volume features (using yesterday's values)
    if 'Volume' in df.columns:
        df['volume_lag1'] = df['Volume'].shift(1)  # Yesterday's volume
        df['volume_lag2'] = df['Volume'].shift(2)  # 2 days ago volume
        df['vol_change_lag1'] = (df['volume_lag1'] / (df['volume_lag2'] + 1e-12) - 1)  # Yesterday's volume change
        df['volume_ma5'] = df['Volume'].rolling(window=5).mean().shift(1)  # Yesterday's 5-day avg volume
        df['volume_ratio'] = df['volume_lag1'] / (df['volume_ma5'] + 1e-12)  # Yesterday's volume vs 5-day avg
    else:
        df['volume_lag1'] = np.nan
        df['volume_lag2'] = np.nan
        df['vol_change_lag1'] = np.nan
        df['volume_ma5'] = np.nan
        df['volume_ratio'] = np.nan
    
    # Momentum features (rate of change)
    df['momentum_5'] = df['Adj Close'].pct_change(periods=5).shift(1)  # 5-day momentum (yesterday)
    df['momentum_10'] = df['Adj Close'].pct_change(periods=10).shift(1)  # 10-day momentum (yesterday)
    
    # Price range features (recent price action)
    df['high_5'] = df['High'].rolling(window=5).max().shift(1)  # Highest high in last 5 days (yesterday)
    df['low_5'] = df['Low'].rolling(window=5).min().shift(1)  # Lowest low in last 5 days (yesterday)
    df['price_vs_range'] = (df['Adj Close'] - df['low_5']) / (df['high_5'] - df['low_5'] + 1e-12)  # Position in 5-day range
    
    # ===== TECHNICAL INDICATORS =====
    df = add_technical_indicators(df)
    
    # ===== MARKET CONTEXT FEATURES =====
    if use_market_context:
        print("[FEATURES] Fetching market context data (SPY, VIX)...")
        start_date = df.index[0] - timedelta(days=60)  # Extra days for rolling calculations
        end_date = df.index[-1] + timedelta(days=1)
        
        market_features = get_market_context_features(start_date, end_date)
        
        # Align market data with stock data
        for key, series in market_features.items():
            if len(series) > 0:
                # Reindex to match stock dates, forward fill missing values
                aligned = series.reindex(df.index).ffill()
                # Backward fill any remaining NaNs at the beginning
                aligned = aligned.bfill()
                df[f'market_{key}'] = aligned
            else:
                df[f'market_{key}'] = np.nan
        
        # Market correlation (rolling correlation with SPY)
        if 'market_spy_return' in df.columns:
            df['market_correlation'] = df['return'].rolling(window=30).corr(df['market_spy_return'])
        
        # Relative strength vs market
        if 'market_spy_return' in df.columns:
            df['relative_strength'] = df['return'] - df['market_spy_return']
    else:
        print("[FEATURES] Skipping market context features")
    
    # ===== OPTIONS FEATURES =====
    if use_options:
        print("[FEATURES] Fetching options data (this may take a while and is limited to recent dates)...")
        print("[FEATURES] Note: yfinance options data is only available for recent dates.")
        
        # Initialize options feature columns
        options_cols = ['iv_mean', 'iv_std', 'atm_iv', 'put_call_volume_ratio', 
                       'put_call_oi_ratio', 'total_options_volume']
        for col in options_cols:
            df[f'options_{col}'] = np.nan
        
        # Try to get options data for recent dates only
        # yfinance typically has options data for the last few months
        recent_dates = df.index[-100000000000000:]  # Last 90 days
        
        options_data_list = []
        for date in recent_dates:
            opt_features = get_options_features(STOCK_TICKER, date)
            if opt_features:
                opt_features['date'] = date
                options_data_list.append(opt_features)
        
        if len(options_data_list) > 0:
            options_df = pd.DataFrame(options_data_list)
            options_df.set_index('date', inplace=True)
            
            # Merge with main dataframe
            for col in options_cols:
                if col in options_df.columns:
                    df.loc[options_df.index, f'options_{col}'] = options_df[col]
            
            # Forward fill options data (use last known value)
            for col in options_cols:
                df[f'options_{col}'] = df[f'options_{col}'].ffill()
            
            print(f"[FEATURES] Added options data for {len(options_df)} dates")
        else:
            print("[FEATURES] No options data available (may be outside yfinance's range)")
    else:
        print("[FEATURES] Skipping options features")
    
    # ===== TIME-BASED FEATURES =====
    df['day_of_week'] = df.index.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df.index.month
    df['is_month_end'] = (df.index.day >= 25).astype(int)  # Approximate month-end effect
    
    # Collect all feature columns
    base_features = [
        # Lagged returns
        'ret_lag1', 'ret_lag2', 'ret_lag3',
        # Price vs Moving Averages
        'price_vs_ma5', 'price_vs_ma10', 'price_vs_ma20',
        # MA Slopes
        'ma5_slope', 'ma10_slope',
        # Volatility
        'vol30_lag1', 'vol10_lag1', 'vol_ratio', 'vol30_lag2',
        # Price Action
        'high_low_range', 'close_position', 'gap',
        # Volume
        'volume_lag1', 'volume_lag2', 'vol_change_lag1', 'volume_ma5', 'volume_ratio',
        # Momentum
        'momentum_5', 'momentum_10',
        # Price Range
        'price_vs_range'
    ]
    
    technical_features = ['rsi', 'macd', 'macd_signal', 'macd_hist', 
                         'bb_width', 'bb_position']
    
    market_features_list = []
    if use_market_context:
        market_features_list = ['market_spy_return', 'market_spy_volatility', 
                               'market_vix', 'market_vix_change',
                               'market_correlation', 'relative_strength']
    
    options_features_list = []
    if use_options:
        options_features_list = [f'options_{col}' for col in 
                                ['iv_mean', 'iv_std', 'atm_iv', 'put_call_volume_ratio',
                                 'put_call_oi_ratio', 'total_options_volume']]
    
    time_features = ['day_of_week', 'month', 'is_month_end']
    
    all_feature_cols = base_features + technical_features + market_features_list + \
                       options_features_list + time_features
    
    # Filter to only existing columns
    feature_cols = [c for c in all_feature_cols if c in df.columns]
    
    # Only drop rows where label is NaN or ALL base features are NaN
    # This is less strict - allows missing market/options data for historical dates
    base_feature_cols = [c for c in base_features if c in df.columns]
    
    # Drop rows where label is missing OR all base features are missing
    if len(base_feature_cols) > 0:
        # Keep rows where at least one base feature is not NaN
        mask = df[base_feature_cols].notna().any(axis=1) & df['label'].notna()
        df_model = df[mask].copy()
    else:
        # Fallback: drop only if label is NaN
        df_model = df.dropna(subset=['label']).copy()
    
    n_after = len(df_model)
    summary = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "n_before": int(n_before),
        "n_after": int(n_after),
        "n_dropped": int(n_before - n_after),
        "num_features": len(feature_cols),
        "features": feature_cols
    }
    
    if os.path.exists(CLEANING_LOG):
        try:
            existing = pd.read_csv(CLEANING_LOG)
            combined = pd.concat([existing, pd.DataFrame([summary])], ignore_index=True)
            combined.to_csv(CLEANING_LOG, index=False)
        except Exception:
            pd.DataFrame([summary]).to_csv(CLEANING_LOG, index=False)
    else:
        pd.DataFrame([summary]).to_csv(CLEANING_LOG, index=False)
    
    print(f"[FEATURES] Feature creation complete.")
    print(f"  Rows: {n_before} -> {n_after} (dropped {n_before - n_after})")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Base features: {len(base_features)}")
    print(f"  Technical indicators: {len(technical_features)}")
    if use_market_context:
        print(f"  Market context: {len(market_features_list)}")
    if use_options:
        print(f"  Options features: {len(options_features_list)}")
    print(f"  Time features: {len(time_features)}")
    
    # Select final columns to save
    out_cols = ['Adj Close', 'return', 'return_next', 'label'] + feature_cols
    if 'Volume' in df_model.columns:
        out_cols.append('Volume')
    
    out_cols = [c for c in out_cols if c in df_model.columns]
    return df_model[out_cols]

def main():
    """Main function to run enhanced feature extraction"""
    ensure_dirs()
    
    print("=" * 70)
    print("ENHANCED FEATURE EXTRACTION")
    print("=" * 70)
    print("This will add:")
    print("  - Technical indicators (RSI, MACD, Bollinger Bands)")
    print("  - Base features (lagged returns, moving averages, volatility)")
    print("Note: Market context (SPY, VIX) and options data downloads are disabled")
    print("=" * 70)
    
    print(f"\n[RUN] Loading raw CSV from: {RAW_CSV}")
    df = load_raw(RAW_CSV)
    
    df = price_consistency_check(df)
    
    # Create enhanced features
    # Options and market context (SPY, VIX) downloads have been disabled
    processed = create_enhanced_features(
        df, 
        use_options=False,  # Disabled: no options data download
        use_market_context=False  # Disabled: no SPY/VIX download
    )
    
    processed.to_csv(PROCESSED_CSV)
    print(f"\n[RUN] Enhanced processed dataset saved to {PROCESSED_CSV} (rows: {len(processed)})")
    print("[RUN] Done.")

if __name__ == "__main__":
    main()

