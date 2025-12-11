# DS-Project
https://youtu.be/ctDz7hkaR00   




How to run:  

CD into "DS-Project"  

If you're on windows, run ".\run_pipeline.bat" and ".\run_tests.bat"  

If you want to use the make file, ensure you are also in the DS-Project directory and run "make run" and "make test" if you wish.  


# Regarding my experiences with the project
In the beginning, I genuinely had no idea what I was doing. I've never tried anything data science-related before, much less something like this. And there is this very cool thing where the less you know about a subject the more confident you think you can pull it off called the Dunning Krueger effect. I want to know what the hell I was smoking when I decided I wanted to try predicting the stock market. I cannot express in words how I feel regarding this. But as silver lining I guess it made me try to think of other ways to improve the project like exploring different tools and methods.     



But let's go back to the beginning. The original goal of this project was to predict the stock market. Thankfully, I got some feedback to classify whether a given stock price goes up, "stays the same", or goes down. But even when I started, given how unexperienced I was (and am), I was kind of just sitting there thinking "Cool. Where do I even start with this?". I eventually got around to grabbing some AAPL stock data from yfinance. Then I extracted some technical features, trained the model and then applied Log Reg, and Random Forest, and then I got a roughly 0.5 average accuracy for the logistic regression and random forest. This was a giant red flag, but I thought I had cooked, and wrote up my midterm report and left it.    

Then the technical midterm rolled around and gave me the biggest reality check. It made me realize I was just drooling all over the keyboard and patting myself on the back for thinking that I could so easily get good accuracy and f1-scores. If I had to suffer through hell just to predict the ratings of music albums, then my results with this project was pulled into question.  

After I came back to this project, I checked for a memory leak and it has turned out the entire time I was predicting whether the next day that AAPL would go up, same, or down, by using data from that exact day. So I was using tomorrow's data to predict tomorrow's results. I had it fixed and it dropped to 0.2-0.3 accuracy and an equally abysmal f1-scoring (which I decided to implement after the midterm). Considering that the dataset we were given for the midterm, my data was insufficient. So what I did was gather options data, general market indicators (the NASDAQ, DOW, and S&P500), and began scraping as much news articles as possible. I was at a lost for finding news articles at first. The most I would be able to get from Google news and was like 100-300 rows, and went back to 2022 at best. But this was clearly not enough for anything. Whether I used those articles for sentiment analysis or not it wouldn't matter.   

So I took to Kaggle and tried there. Most datasets I found were limited by the amount of years covered, so I kept them for the time being. I decided to exclude Kaggle from my search terms and eventually found my goat the FNSPID dataset (https://github.com/Zdong104/FNSPID_Financial_News_Dataset) with 30+GB of financial news articles. With this I left my computer runnig overnight and extracted 6GB worth of news articles relating to AAPL from 1999 to 2024. I later used these in conjunction with more recent articles. Then I used VADER sentiment analysis, TF-IDF to create sentiment analysis features.   

So when I used all this new data, I found that both options data, and the market indicators actually ended up decreasing my scores. Why? Options data were limited and I couldn't obtain options data beyond 90ish days, let alone years. But with the market indicators it is due to the fact that they represent the entire market and thus take into account many other stocks. This added a lot of noise to my features and ended up dragging the entire thing as a whole. It would be better to just focus on AAPL by itself since Apple is such a massive company that things just work differently for them compared to the majority of the other stocks. Or just skill issue. But at any rate, I didn't find success with those, so I turned to sentiment analysis. I tried implementing what I learned from the midterm and implemented Vader wit TF-IDF, with an extensive word bank, and saw improvements to my scoring. 
But let's discuss the things after data collection and cleaning. With features I decided I didn't have enough and added more specific finance related features like MACD and Bollinger Bonds and the like. In addition, I ended up using multiple other models, specifically CatBoost, XGBoost, and LSTMs. However I found that Catboost was always underperforming and was dead last when it came to performance, so it was removed. In hindsight I believe it would've been better to just focus on logistic regression and xgboost only since they gave me the best initial performances, and by focusing on them I could've spent more time on refining them. I used hyper tuning but perhaps I could've done more.   

I ended up changing the up, same, and down classification to just up and down. I found it difficult to determine the correct delta thresholds, in addition to creating a wildly imbalanced same class that I just couldn't seem to address even when I tried to balance it out, it seemed the models had extreme difficult with classifying them. It was either all in same or all in up/down. So I got rid of it. Now I had a new problem. Turns out everything was being dumped into either down or up. At this point I just had to accept the futility of predicting the stock market. I looked online and this is the focal point of actual professionals. If they could crack this they'd be billionaires, so who am I to do such things when I have little to no experience?   

So I just moved onto the trading aspect. A bit unsurprisingly, it turned out just buying AAPL stock and then holding it for several years ended up with a return of 300%, while my logistic regression model only netted a 270% return over the same period. 

  
In short, I learned a lot about the data science cycle, but it had to be through an impossible task because I had no idea what I was getting myself into.


  
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
