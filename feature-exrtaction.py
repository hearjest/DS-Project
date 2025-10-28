import os
import pandas as pd
import numpy as np
from datetime import datetime


RAW_CSV = "test4.csv"
PROCESSED_CSV = "processedtest4.csv"
CLEANING_LOG = "cleaning_log.csv"
DELTA = 0.003        
EXTREME_THRESHOLD = 0.10 
# -----------------------

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

def load_raw(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw CSV not found: {path}. Run get_data.py or rename your CSV accordingly.")
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    try:
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    except Exception:
        df.index = pd.to_datetime(df.index).tz_localize(None, ambiguous='infer', nonexistent='shift_forward')
    df = df.sort_index()
    return df

def price_consistency_check(df):
    # Check logical OHLC relations: Low <= Open/Close <= High
    cond = (
        (df['Open'] > df['High']) |
        (df['Close'] > df['High']) |
        (df['Low'] > df['Open']) |
        (df['Low'] > df['Close'])
    )
    violations = df[cond]
    if len(violations) > 0:
        violations.reset_index().to_csv("data/price_consistency_violations.csv", index=False)
        df = df.drop(index=violations.index)
        print(f"[CLEAN] Price consistency: dropped {len(violations)} rows (saved to data/price_consistency_violations.csv).")
    else:
        print("[CLEAN] Price consistency: no violations found.")
    return df

def corporate_actions_check(df):
    if 'Adj Close' not in df.columns:
        print("[CLEAN] Warning: 'Adj Close' column not found. Returns may not reflect corporate actions.")
    else:
        # Log nonzero splits/dividends if present
        if 'Stock Splits' in df.columns:
            splits = df[df['Stock Splits'] != 0][['Stock Splits']]
            if len(splits) > 0:
                splits.reset_index().to_csv("data/splits_logged.csv", index=False)
                print(f"[CLEAN] Logged {len(splits)} stock split rows to data/splits_logged.csv")
            else:
                print("[CLEAN] No stock splits found (or all zeros).")
        else:
            print("[CLEAN] 'Stock Splits' column not present in raw CSV.")
        if 'Dividends' in df.columns:
            divs = df[df['Dividends'] != 0][['Dividends']]
            if len(divs) > 0:
                divs.reset_index().to_csv("data/dividends_logged.csv", index=False)
                print(f"[CLEAN] Logged {len(divs)} dividend rows to data/dividends_logged.csv")
            else:
                print("[CLEAN] No dividends found (or all zeros).")
        else:
            print("[CLEAN] 'Dividends' column not present in raw CSV.")

def outlier_detection(df):
    # After returns computed, flag extreme daily moves
    extreme = df[df['return'].abs() > EXTREME_THRESHOLD][['return']].copy()
    if len(extreme) > 0:
        df.loc[extreme.index].reset_index().to_csv("data/extreme_days_fullrows.csv", index=False)
        print(f"[CLEAN] Flagged {len(extreme)} extreme-return days (>|{EXTREME_THRESHOLD*100:.0f}%|) -> data/extreme_days_fullrows.csv")
    else:
        print(f"[CLEAN] No extreme-return days above |{EXTREME_THRESHOLD*100:.0f}%|")
    top_pos = df['return'].nlargest(20)
    top_neg = df['return'].nsmallest(20)
    pd.DataFrame({'date': top_pos.index, 'return': top_pos.values}).to_csv("data/top5_positive_returns.csv", index=False)
    pd.DataFrame({'date': top_neg.index, 'return': top_neg.values}).to_csv("data/top5_negative_returns.csv", index=False)

def label_return(r, delta=DELTA):
    if pd.isna(r):
        return np.nan
    if r > delta:
        return "Up"
    elif r < -delta:
        return "Down"
    else:
        return "Same"

def create_features(df):
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']

    df['return'] = df['Adj Close'].pct_change()
    df['label'] = df['return'].apply(lambda x: label_return(x))

    outlier_detection(df)

    n_before = len(df)

    df['ret_lag1'] = df['return'].shift(1)
    df['ret_lag2'] = df['return'].shift(2)
    df['ma5'] = df['Adj Close'].rolling(window=5).mean().pct_change()
    df['ma10'] = df['Adj Close'].rolling(window=10).mean().pct_change()
    df['vol30'] = df['return'].rolling(window=30).std()
    df['vol10'] = df['return'].rolling(window=10).std()
    df['vol_ratio'] = df['vol10'] / (df['vol30'] + 1e-12)
    df['vol_lag1'] = df['vol30'].shift(1)
    df['vol_lag2'] = df['vol30'].shift(2)
    df['volume_lag1'] = df['Volume'].shift(1) if 'Volume' in df.columns else np.nan
    df['vol_change'] = df['Volume'] / (df['volume_lag1'] + 1e-12) - 1 if 'Volume' in df.columns else np.nan

    feature_cols = [
        'ret_lag1','ret_lag2','ma5','ma10','vol30','vol10',
        'vol_ratio','vol_lag1','vol_lag2','vol_change'
    ]
    df_model = df.dropna(subset=feature_cols + ['label']).copy()


    n_after = len(df_model)
    summary = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "n_before": int(n_before),
        "n_after": int(n_after),
        "n_dropped": int(n_before - n_after)
    }
    if os.path.exists(CLEANING_LOG):
        try:
            existing = pd.read_csv(CLEANING_LOG)
            combined = pd.concat([existing, pd.DataFrame([summary])], ignore_index=True)
            combined.to_csv(CLEANING_LOG, index=False)
        except Exception:
            # if read fails, overwrite with fresh log
            pd.DataFrame([summary]).to_csv(CLEANING_LOG, index=False)
    else:
        pd.DataFrame([summary]).to_csv(CLEANING_LOG, index=False)
    print(f"[CLEAN] Feature creation complete. Rows before features: {n_before}, after dropna: {n_after} (dropped {n_before - n_after}). Logged to {CLEANING_LOG}")

    # Select final columns to save (date index will be preserved by to_csv)
    out_cols = ['Adj Close', 'return', 'label'] + [c for c in feature_cols if c in df_model.columns] + (['Volume'] if 'Volume' in df_model.columns else [])
    out_cols = [c for c in out_cols if c in df_model.columns]
    return df_model[out_cols]

def main():
    ensure_dirs()
    print(f"[RUN] Loading raw CSV from: {RAW_CSV}")
    df = load_raw(RAW_CSV)

    df = price_consistency_check(df)

    corporate_actions_check(df)

    processed = create_features(df)

    processed.to_csv(PROCESSED_CSV)
    print(f"[RUN] Processed dataset saved to {PROCESSED_CSV} (rows: {len(processed)})")
    print("[RUN] Done.")

if __name__ == "__main__":
    main()
