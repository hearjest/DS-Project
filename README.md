# DS-Project



Updated for the final report.

# Data Collection
I used the yfinance API to grab stock data (and options) about AAPL from the very beginning. I've also grabbed data from the NASDAQ, DOW, and S&P500. These and the options data were not used in the end. Explaination following after the model training section.
In addition I used scrapers to scrape headlines involving Apple stock using NEWSAPI from Google News. In addition, I've used multiple Kaggle datasets also containing news about Apple. The biggest dataset by far was the FNSPID dataset (https://github.com/Zdong104/FNSPID_Financial_News_Dataset). It contains roughly 33GB of scraped news articles from 1999 to 2024. This was later combined with articles from Alpha Vantage, the scraped Apple articles. 

# Data Cleaning
The only real cleaning I had to do was filter through the 33GB dataset to filter for articles that involve Apple stock, AAPL, etc.. I was then able to obtain roughly 6-7 GB of articles. 


# Feature Extraction
Probably the worst part. I used many financial metrics such as volume, rolling averages

# Data Visualization

# Model Training

# Apple Stock Price Prediction Project

## Project Overview

This project implements a comprehensive machine learning pipeline to predict next-day directional movements (Up/Down) for Apple Inc. (AAPL) stock. The project integrates historical stock price data, news sentiment analysis, technical indicators, and multiple classification models to demonstrate the full data science lifecycle.

The goal is to practice the full data science lifecycle on a topic of your choice, and this project includes all required components:

- **Data Collection**
- **Data Cleaning**
- **Feature Extraction**
- **Data Visualization**
- **Model Training**

---

## 1. Data Collection

### 1.1 Stock Price Data
- **Source**: Yahoo Finance via `yfinance` library
- **Period**: 1996 to October 2025 (~29 years of data, ~7,500 trading days)
- **Script**: `getData.py`
- **Data Collected**:
  - OHLCV (Open, High, Low, Close, Volume)
  - Adjusted Close (auto-adjusted for splits/dividends)
  - Dividends and stock splits information
- **Output**: `test4.csv`, `test5.csv`

### 1.2 News Articles Data
- **Sources**:
  - Google News RSS feeds
  - NewsAPI
  - External datasets (NASDAQ, external news sources)
- **Scripts**:
  - `Data collection/scrape_apple_news.py`: Scrapes Google News and NewsAPI
  - `Data collection/extract_apple_articles.py`: Filters Apple-related articles from external datasets
- **Filtering**:
  - Keyword-based detection (AAPL, Apple Inc., product mentions, financial terms)
  - Duplicate removal (URL and title+date based)
- **Output**: `apple_articles_compiled2.csv`

### 1.3 Sentiment Data
- **Methods**:
  1. VADER Sentiment Analysis (`Feature extraction/analyze_sentiment_vader.py`)
  2. Naive Bayes + TF-IDF (`Feature extraction/analyze_sentiment_nb_tfidf.py`)
- **Outputs**:
  - `apple_daily_sentiment.csv` (VADER-based daily sentiment)
  - `apple_daily_sentiment_nb.csv` (Naive Bayes daily sentiment)

---

## 2. Data Cleaning

### 2.1 Price Data Cleaning
**Scripts**: `Feature extraction/feature-exrtaction.py`, `feature_extraction_enhanced.py`

#### Price Consistency Checks
- Validates logical OHLC relationships:
  - Low ≤ Open/Close ≤ High
  - Identifies and removes violations
- **Logging**: Violations saved to `data/price_consistency_violations.csv`
- **Result**: 4 rows removed due to price inconsistencies

#### Missing Data Handling
- Forward-fills missing values where appropriate
- Drops rows with insufficient data for rolling features (32 rows removed at start due to monthly feature requirements)
- **Result**: 7,546 → 7,514 rows after cleaning

#### Corporate Actions Tracking
- Logs stock splits (`data/splits_logged.csv`)
- Logs dividend payments (`data/dividends_logged.csv`)
- Uses `auto_adjust=True` in yfinance to normalize prices

#### Outlier Detection
- Flags extreme return days (threshold: 25% daily move)
- Logs extreme days to `data/extreme_days_fullrows.csv`
- Logs top 20 positive and negative returns
- **Note**: Outliers are retained intentionally to reflect market realities (bubbles, crises, volatility events)

### 2.2 News Article Cleaning
- Text preprocessing:
  - Lowercasing
  - URL/email removal
  - Special character removal
  - Whitespace normalization
- Date normalization:
  - Timezone handling
  - Date parsing with error handling

### 2.3 Sentiment Data Aggregation
- Daily aggregation of article-level sentiment
- Handles missing dates
- Computes daily statistics (mean, std, counts)

---

## 3. Feature Extraction

### 3.1 Technical Indicators
**Scripts**: `Feature extraction/feature-exrtaction.py`, `feature_extraction_enhanced.py`

#### Price-Based Features
- **Returns**: Daily percentage change in adjusted close
- **Lagged Returns**: `ret_lag1`, `ret_lag2`, `ret_lag3`
- **Moving Averages**: 5-day, 10-day, 20-day
- **Price vs MA**: Percentage above/below MAs (`price_vs_ma5`, `price_vs_ma10`, `price_vs_ma20`)
- **MA Slopes**: Rate of change in moving averages

#### Volatility Features
- **Rolling Volatility**: 10-day and 30-day standard deviation of returns
- **Volatility Ratios**: Short-term vs long-term (`vol_ratio`)
- **Lagged Volatility**: `vol30_lag1`, `vol30_lag2`

#### Volume Features
- **Lagged Volume**: `volume_lag1`, `volume_lag2`
- **Volume Change**: Day-over-day volume change
- **Volume Ratios**: Current vs 5-day average volume

#### Price Action Features
- **High-Low Range**: Daily range as percentage of price
- **Close Position**: Where close falls within daily range (0-1)
- **Gap**: Opening gap from previous close
- **Momentum**: 5-day and 10-day price momentum
- **Price vs Range**: Position within 5-day high/low range

#### Technical Indicators (Enhanced Version)
- **RSI** (Relative Strength Index): 14-day momentum oscillator
- **MACD**: Moving Average Convergence Divergence with signal line
- **Bollinger Bands**: Volatility bands with width and position indicators

### 3.2 Sentiment Features
#### VADER Sentiment Features
- `VADER_Compound_Mean`: Daily average compound sentiment score
- `VADER_Compound_Std`: Standard deviation of compound scores
- `VADER_Positive_Mean`: Average positive sentiment
- `VADER_Negative_Mean`: Average negative sentiment
- `VADER_Neutral_Mean`: Average neutral sentiment

#### Word Bank Features
- `WordBank_Positive_Sum`: Count of positive financial keywords
- `WordBank_Negative_Sum`: Count of negative financial keywords
- `WordBank_Net_Sentiment`: Net sentiment (positive - negative)
- Word bank includes 100+ financial terms (earnings, growth, analyst ratings, etc.)

#### Naive Bayes Sentiment Features
- `NB_Positive_Ratio`: Ratio of positive articles
- `NB_Negative_Ratio`: Ratio of negative articles
- `NB_Net_Sentiment`: Net sentiment count
- `NB_Net_Sentiment_Ratio`: Normalized net sentiment

### 3.3 Market Context Features (Optional)
- **SPY Returns**: S&P 500 ETF returns (market proxy)
- **SPY Volatility**: Market volatility measure
- **VIX**: Volatility index (fear gauge)
- **Market Correlation**: Rolling correlation with SPY
- **Relative Strength**: Stock return minus market return

### 3.4 Time-Based Features
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `month`: Month of year (1-12)
- `is_month_end`: Month-end indicator

### 3.5 Target Variable Creation
- **Binary Classification**: "Up" vs "Down"
- **Delta Threshold**: 0.4% (return > 0.004 = "Up", else "Down")
- **Labels Shifted Forward**: Predicts next day's movement
- **Result**: 50 features used for model training

---

## 4. Data Visualization

### 4.1 Stock Price Visualizations
**Script**: `Data visualization/visuals.py`, `s3.4testsentiment2.ipynb`

#### Time Series Plots
- **Adjusted Close Price**: Full historical price series (1996-2025)
- **Rolling Volatility**: 30-day annualized volatility over time
- Highlights major events: Dot-com bubble (2000), financial crisis (2008), COVID-19 (2020)

#### Distribution Plots
- **Daily Returns Histogram**: Distribution of daily percentage returns
- **Class Distribution**: Bar chart showing Up/Down class counts

### 4.2 Sentiment Visualizations
**Script**: `Feature extraction/analyze_sentiment_vader.py`

#### Sentiment Time Series
- **VADER Compound Sentiment**: Mean ± 1 standard deviation over time
- **VADER Breakdown**: Positive, negative, and neutral sentiment trends
- **Word Bank Net Sentiment**: Positive minus negative word counts
- **Word Bank Counts**: Positive vs negative word counts over time
- **Article Count**: Number of articles per day

#### Comparison Plots
- **VADER vs Word Bank**: Normalized comparison of both sentiment methods

### 4.3 Model Performance Visualizations
**Notebook**: `s3.4testsentiment2.ipynb`

#### Confusion Matrices
- Individual confusion matrices for each model
- Combined comparison view
- Saved to: `plots/confusion_matrix_*.png`

#### Trading Performance
- Portfolio value over time for each model
- Comparison vs buy-and-hold strategy
- Trade frequency visualization
- Saved to: `plots/trading_performance.png`

### 4.4 Feature Analysis Visualizations
- Feature importance plots (Random Forest, XGBoost)
- Delta threshold analysis: Impact of different delta values on class distribution
- Correlation heatmaps (optional)

---

## 5. Model Training

### 5.1 Data Splitting
- **Time-based Split**: 80% training (1996-2020), 20% testing (2020-2025)
- **No Random Shuffling**: Preserves temporal order
- **Training Set**: ~6,000 days
- **Test Set**: ~1,500 days

### 5.2 Feature Scaling
- **StandardScaler**: Standardizes features (mean=0, std=1)
- **Fit on Training Data Only**: Applied to test data

### 5.3 Models Implemented
**Script**: `Model training/model-train.py`, `s3.4testsentiment2.ipynb`

#### 1. Logistic Regression
- Multi-class (multinomial) logistic regression
- Class weight balancing for imbalanced classes
- Max iterations: 2,000
- **Test Accuracy**: **53.47%**
- **Precision/Recall**: Down (63.5%/46.7%), Up (46.0%/62.8%)

#### 2. Random Forest
- Ensemble of 200 decision trees
- Class weight balancing
- **Test Accuracy**: **48.33%**
- **Precision/Recall**: Down (57.5%/42.0%), Up (41.6%/57.1%)
- Feature importance analysis available

#### 3. XGBoost
- Gradient boosting with regularization
- Hyperparameter tuning available (disabled by default)
- **Test Accuracy**: **51.27%**
- **Precision/Recall**: Down (62.0%/41.6%), Up (44.4%/64.7%)
- Feature importance analysis available

#### 4. PyTorch Neural Network
- LSTM-based architecture for time series
- Architecture: LSTM layers → Fully connected layers → Softmax
- CUDA support if available
- **Test Accuracy**: **46.27%**
- **Precision/Recall**: Down (56.3%/33.4%), Up (41.0%/64.1%)

#### 5. Naive Bayes (Sentiment Classification)
- MultinomialNB with TF-IDF features
- Used for news sentiment classification
- Trained on VADER-labeled articles
- Binary classification: Positive vs Negative sentiment

### 5.4 Model Evaluation Metrics
**Script**: `s3.4testsentiment2.ipynb`

#### Classification Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: Per-class precision
- **Recall**: Per-class recall
- **F1-Score**: Per-class F1-score
- **Macro F1**: Average F1 across classes
- **ROC-AUC**: Area under ROC curve (binary)

#### Trading Performance Metrics
- **Total Return**: Portfolio return over test period
- **Buy-and-Hold Return**: Baseline comparison
- **Excess Return**: Model return minus buy-and-hold
- **Number of Trades**: Trade frequency
- **Sharpe Ratio** (optional): Risk-adjusted return

### 5.5 Model Results Summary
**Best performing model**: **Logistic Regression** (53.47% accuracy)

| Model | Test Accuracy | Macro F1 | Notes |
|-------|---------------|----------|-------|
| Logistic Regression | 53.47% | 0.5346 | Best overall performance |
| XGBoost | 51.27% | 0.5122 | Moderate performance |
| Random Forest | 48.33% | 0.4833 | Below baseline |
| PyTorch LSTM | 46.27% | 0.4597 | Underperforms (possible overfitting) |

**Note**: Stock price prediction is inherently challenging; accuracy above 50% indicates some predictive signal.

### 5.6 Model Persistence
- Models saved to: `models/*.pkl`
- Includes: Model, scaler, feature columns
- Loading: `joblib.load()`

---

## Project Structure

```
DS-Project/
├── Data collection/
│   ├── scrape_apple_news.py          # News scraping from Google/NewsAPI
│   ├── extract_apple_articles.py     # Filter Apple articles from datasets
│   └── validate_compiled_articles.py # Article validation
├── Feature extraction/
│   ├── feature-exrtaction.py         # Base feature extraction
│   ├── feature_extraction_enhanced.py # Enhanced features (technical indicators)
│   ├── analyze_sentiment_vader.py    # VADER sentiment analysis
│   └── analyze_sentiment_nb_tfidf.py # Naive Bayes sentiment classification
├── Data visualization/
│   └── visuals.py                    # Visualization functions
├── Model training/
│   └── model-train.py                # Model training script
├── models/                           # Saved models
├── plots/                            # Generated visualizations
├── results/                          # Metrics and reports
├── data/                             # Intermediate data files
├── s3.4testsentiment2.ipynb         # Main notebook (complete pipeline)
├── getData.py                        # Stock data collection
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Key Findings

1. **Data Quality**: Successfully processed ~7,500 trading days with 99.6% data retention after cleaning
2. **Feature Engineering**: Created 50+ features including technical, sentiment, and time-based features
3. **Sentiment Integration**: Integrated VADER and Naive Bayes sentiment analysis from 1,000+ daily articles
4. **Model Performance**: Logistic Regression achieves 53.47% accuracy (above random chance)
5. **Challenges**: Stock prediction is inherently difficult; models struggle with class imbalance and market efficiency

---

## Dependencies

See `requirements.txt` for full list. Key packages:
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: Machine learning
- `xgboost`: Gradient boosting
- `torch`: Deep learning
- `yfinance`: Stock data
- `vaderSentiment`: Sentiment analysis
- `matplotlib`: Visualization

---

## Usage Instructions

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Collect stock data**: `python getData.py`
3. **Collect news data**: Run scripts in `Data collection/`
4. **Extract features**: Run `feature_extraction_enhanced.py`
5. **Analyze sentiment**: Run scripts in `Feature extraction/`
6. **Train models**: Run `s3.4testsentiment2.ipynb` or `Model training/model-train.py`
7. **View results**: Check `results/` and `plots/` directories

---

## Future Improvements

1. **Feature Engineering**: Additional technical indicators, market microstructure features
2. **Sentiment Analysis**: Advanced NLP models (BERT, transformer-based)
3. **Model Tuning**: Hyperparameter optimization, ensemble methods
4. **External Data**: Options data, social media sentiment, earnings data
5. **Evaluation**: More robust backtesting, transaction costs, risk metrics

---

This project demonstrates a complete data science lifecycle for financial prediction, integrating multiple data sources and modeling approaches to predict stock price movements.
