# Introducing Crypto Price Predictor
![Crypto Predictor UI](images/Screenshot%202025-08-26%20at%2000.41.28.png)


# Overview
A simple web application that predicts the future price of cryptocurrencies using
a machine learning model.

## How It Works

This application uses an XGBoost Regressor model to predict cryptocurrency
prices. The model is trained on historical daily price data (Open, High, Low,
Close, Volume) for the past year, fetched from Yahoo Finance.

The prediction is based on the following features:

- 7-day Simple Moving Average (SMA)
- 14-day Simple Moving Average (SMA)
- 4-day Price Momentum
- 7-day Price Volatility

The model makes a base prediction, which is then adjusted by projecting the
recent daily return trend into the future.

## Features

- Web Interface: A simple Flask web interface to interact with the model.
- Dynamic Data: Fetches the latest cryptocurrency data from Yahoo Finance.
- On-the-fly Training: The model is trained with the latest data when the
  application starts.
- Price Prediction: Predicts the price for a given cryptocurrency for a
  specified number of days in the future (1-90).

## Dependencies

The project's core dependencies are:

- Flask: For the web application. <3
- yfinance: To download historical market data from Yahoo Finance.
- pandas: For data manipulation and analysis. <3
- scikit-learn: For data preprocessing (MinMaxScaler). <3
- xgboost: For the prediction model (XGBRegressor).

You can install these dependencies using the provided `requirements.txt` file.

## Installation

1. **Clone the repository:**

   ```bash
   git clone [insert Repo-URL here]
   cd crypto-predictor
   ```

2. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Flask application:**

   ```bash
   python src/app.py
   ```

2. **Open your web browser** and navigate to:

   ```text
   http://127.0.0.1:5000
   ```

3. **Enter a cryptocurrency symbol** (e.g., `BTC`, `ETH`, `SOL`) and the number
   of days you want to predict into the future.

4. **Click "Predict Price"** to see the result.

## License

This project is protected under MIT License.

## Disclaimer

This project is for educational purposes only. The predictions are based on a
simple model and historical data, and should not be considered financial advice.
Cryptocurrency markets are highly volatile, and you should do your own research
before making any investment decisions.
