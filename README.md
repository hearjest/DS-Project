# DS-Project

Description: This project will predict the next day's trading price of a selected stock(s). This will be done by factoring in multiple factors, such as the stock's historical performance, options data, and data from the rest of the stock market (including the 2008 market crash and 2020).

Goals: Accurately predict the next day's stock price, take appropriate actions based on the prediction, and yield a net profit in the end. In addition, the net profit should be greater than that of buying and holding.

Required Data: 
- Historical Stock Price Data (daily opens, highs, lows, close, volume, etc. for as long as possible back) for selected stock and other points of comparison, such as the S&P 500. This data can be grabbed from the Alpha Vantage API and yfinance.
- Options market data will be also grabbed from the Polygon.io api

Data Modeling Methods: Fitting linear model and decision trees

Data Visualization Methods: Line chart illustrating the differences in net profit earned between buying and holding and next-day predictions.

Test Plans: Not sure how far back all the data goes back, but I plan on using 80% of the data for training, the next 10% to check it works, and the remaining 10% will be used for testing.
