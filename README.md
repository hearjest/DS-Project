https://youtu.be/sHa6sL3icTg 

WARNING: THERE WILL BE STATIC THROUGHOUT THE VIDEO SO DON'T WEAR HEADPHONES

# DS-Project

Updated for midterm report

Description: This project will predict the next day's trading price of a selected stock(s). This will be done by factoring in multiple factors, such as the stock's historical performance, and data from the rest of the stock market.

# Preliminary Visualizations of Data:
<img width="1000" height="400" alt="Figure_1" src="https://github.com/user-attachments/assets/6632681a-9923-409b-b8cd-c9883abf9bd4" />
<img width="800" height="400" alt="Figure_2" src="https://github.com/user-attachments/assets/25057d6c-6051-4833-80cc-72151fe09ce5" />
<img width="1000" height="400" alt="Figure_3" src="https://github.com/user-attachments/assets/706ca83d-af51-4f46-98f6-cd3cd958c588" />
<img width="800" height="400" alt="Figure_4" src="https://github.com/user-attachments/assets/a04eb957-b9ef-4716-a3d6-c77031dd5cef" />



# Data Processing:
Notice the volatility during the Dot Com Bubble, 2008 Financial Crisis, Covid, and Trump annoucing tariff pauses in the 3rd image. I've decided to keep these in because I wanted to account for any pattern to all of these events occuring, causing high volatility, even if I cannot predict them. I don't think it would reflect reality, which is what we want to reflect, if we continually assume controlled and stable growth/stagnation.

I sourced my data from yfinance, an API that uses Yahoo Finance's APIs to collect all stock data from AAPL from 1996 until Mid October 2025. I chose to use auto_adjust set to True because I wanted the data to not be muddied by Apple doing stuff with splits and dividends. 

I dropped 32 rows from the beginning because a feature was monthly based and since it required a month, the dataframe had nulls in it.

4 rows got flagged for price inconsistency and since 1-3 of the last digits of the closing prices were somehow greater than of the high prices. I believe this is most likely due to miniscule rounding. 

# Initial Data Modeling Methods

I used 80% of the data for training, and used the remaining 20% for testing.

I did not know what data modeling methods to use from lecture so I looked them up. I used Logistic Regression and Random Forest. I used Logistic Regression because on the feedback, I was told it would be easier if I predicted whether the stock price went up, down or stayed the same rather than an exact number (and I agree, that's too ambitious) so I went with a classification kind of goal. Since logistic regression is made for classification, and seemed to be mentioned a lot, I thought it was a good fit. This is in addition to it being fast.
Random Forest was also chosen because it can handle complex data and I could see how having multiple features to each tree could work in my favor to more accurately which class.

# Preliminary Results

<img width="308" height="158" alt="Screenshot 2025-10-27 221149" src="https://github.com/user-attachments/assets/18bf2f68-a610-4491-b32e-fb1e9ed23a7b" />
<img width="486" height="408" alt="Screenshot 2025-10-27 221224" src="https://github.com/user-attachments/assets/48b092b9-2a26-455a-9529-ff7d3c8f4bea" />

I was pretty happy to be able to get 60% with logistic regression. But the recall score is absolutely abysmal and needs to be addressed, and just the entire Same classification as well.


<img width="565" height="318" alt="Screenshot 2025-10-27 221253" src="https://github.com/user-attachments/assets/04a33073-ccb3-4d78-a291-018e1e5b9863" />
<img width="497" height="412" alt="Screenshot 2025-10-27 221317" src="https://github.com/user-attachments/assets/4a5daba0-aec0-4ed4-a290-a9faa3ccaae9" />

I was surprised to see a 57% accuracy score from random forest, and we see another abysmal Same classification score. This needs to be addressed.

# Next Steps / Plans for final 
I think I'm off to a good start but there needs to be some changes. I'm starting to doubt some of the features I'm using such as the averages because of the outliers that I intentionally kept. I will also account for the overall performance of the stock market, and try to include some of the options data to see if it'll help. I also think I need to test with a few other classification models. 













# Old / Proposal

Description: This project will predict the next day's trading price of a selected stock(s). This will be done by factoring in multiple factors, such as the stock's historical performance, options data, and data from the rest of the stock market (including the 2008 market crash and 2020).
**Post Feedback** - I will be using the AAPL (Apple) stock primarily. I will not be using the options market, it'll be a bonus implementation.

Goals: Accurately predict the next day's stock price, take appropriate actions based on the prediction, and yield a net profit in the end. In addition, the net profit should be greater than that of buying and holding.
**Post Feedback** - Instead of predicting solid numbers, I will instead only predict whether the price goes up, down, or stays relatively the same. I did have any other metrics besides getting a net profit and beating buy and hold strategy, so I will add: percentage of predictions being correct in terms of whether the stock increased, or decreased, and total amount of money lost and gained.

Required Data: 
- Historical Stock Price Data (daily opens, highs, lows, close, volume, etc. for as long as possible back) for selected stock and other points of comparison, such as the S&P 500. This data can be grabbed from the Alpha Vantage API and yfinance.
- Options market data will be also grabbed from the Polygon.io api

Data Modeling Methods: Fitting linear model and decision trees

Data Visualization Methods: Line chart illustrating the differences in net profit earned between buying and holding and next-day predictions.

Test Plans: Not sure how far back all the data goes back, but I plan on using 80% of the data for training, the next 10% to check it works, and the remaining 10% will be used for testing. 
