import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

# === User Input ===
ticker_base = input("Enter NSE stock symbol (example: RELIANCE): ").strip().upper()
if not ticker_base.endswith('.NS'):
    ticker = ticker_base + '.NS'
else:
    ticker = ticker_base

purchase_date = input("Enter Date of Purchase (YYYY-MM-DD): ")
future_date = input("Enter Future Date to Predict Price (YYYY-MM-DD): ")

# === Get stock listing date dynamically ===
ticker_obj = yf.Ticker(ticker)
hist = ticker_obj.history(period="max")
if hist.empty:
    print(f"No historical data found for {ticker}. Check symbol and try again.")
    exit()
listing_date = hist.index.min().date()
print(f"Detected listing date for {ticker}: {listing_date}")

# === Download stock data from listing date ===
data = yf.download(ticker, start=listing_date)
series = data['Close'].dropna()
series.index = pd.DatetimeIndex(series.index)
series = series.asfreq('B')               # Business day frequency
series = series.fillna(method='ffill')    # Fill missing days
today = series.index[-1]

# === Adjust purchase date to nearest trading day ===
purchase_date = pd.to_datetime(purchase_date)
while purchase_date.strftime("%Y-%m-%d") not in series.index.strftime("%Y-%m-%d"):
    purchase_date += BDay(1)
purchase_date_str = purchase_date.strftime("%Y-%m-%d")
purchase_price = float(series.loc[purchase_date_str])

# === Adjust future date to next trading day if needed ===
future_date = pd.to_datetime(future_date)
if future_date <= today:
    print("Future date must be greater than latest available stock date.")
    exit()
while future_date.weekday() >= 5:
    future_date += BDay(1)

# === Fetch financial data ===
balance_sheet = ticker_obj.balance_sheet.T
income_statement = ticker_obj.financials.T
exog_df = pd.concat([balance_sheet, income_statement], axis=1)
exog_df = exog_df.reindex(series.index, method='ffill').fillna(method='ffill')

# Select relevant financial features
features = [
    'Total Assets', 'Total Current Liabilities', 'Total Liab',
    'Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income'
]
existing_features = [f for f in features if f in exog_df.columns]
exog_aligned = exog_df[existing_features].copy()

# Clean exog variables to remove NaNs and infs
exog_aligned = exog_aligned.apply(pd.to_numeric, errors='coerce')
exog_aligned = exog_aligned.dropna(axis=1, how='all')
exog_aligned = exog_aligned.fillna(method='ffill').fillna(method='bfill')
exog_aligned = exog_aligned.replace([np.inf, -np.inf], 0)
exog_aligned = exog_aligned.dropna(axis=0, how='any')

# Align series with exogenous data after cleaning
series = series.loc[exog_aligned.index]
today = series.index[-1]

# === Set forecast horizon ===
max_days = 180  # adjust max forecast length
end_forecast_date = min(today + BDay(max_days), future_date)
forecast_index = pd.bdate_range(today + BDay(1), end_forecast_date)
days_to_forecast = len(forecast_index)

if days_to_forecast == 0:
    print("Forecast date too close or no business days available.")
    exit()

# === Model selection ===
if days_to_forecast <= 60:
    print("Using ARIMA for short-term forecast.")
    model = pm.auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True)
    arima_model = ARIMA(series, order=model.order)
    fit_model = arima_model.fit()
    forecast = fit_model.forecast(steps=days_to_forecast)
else:
    print("Using SARIMAX for long-term forecast with exogenous financials.")
    model = pm.auto_arima(series, exogenous=exog_aligned, seasonal=True, m=5,
                          stepwise=True, suppress_warnings=True)
    seasonal_order = model.seasonal_order if hasattr(model, 'seasonal_order') else (0, 0, 0, 0)
    sarima_model = SARIMAX(series,
                           order=model.order,
                           seasonal_order=seasonal_order,
                           exog=exog_aligned)
    fit_model = sarima_model.fit(disp=False)
    exog_future = exog_aligned.iloc[-days_to_forecast:]
    forecast = fit_model.forecast(steps=days_to_forecast, exog=exog_future)

forecast_series = pd.Series(forecast, index=forecast_index)
future_target_date = forecast_index[-1]
future_price = float(forecast_series.iloc[-1])

if pd.isna(future_price):
    print("Forecast failed. Try a closer future date or more historical data.")
    exit()

latest_price = float(series.iloc[-1])
profit_loss_purchase_to_future = future_price - purchase_price

# === Print Summary ===
print(f"\n--- Price Summary for {ticker} ---")
print(f"Purchase Date (Adjusted): {purchase_date_str}, Price: ₹{purchase_price:.2f}")
print(f"Latest Date: {today.date()}, Price: ₹{latest_price:.2f}")
print(f"Profit/Loss (Purchase → Today): ₹{latest_price - purchase_price:.2f}")
print(f"Future Date (Adjusted): {future_target_date.date()}, Predicted Price: ₹{future_price:.2f}")
print(f"Profit/Loss (Today → Future): ₹{future_price - latest_price:.2f}")
print(f"Profit/Loss (Purchase → Future): ₹{profit_loss_purchase_to_future:.2f}")

# === Plotting ===
plt.figure(figsize=(14, 7))
plt.plot(series.index, series, label='Historical Price', color='blue')
plt.plot(forecast_series.index, forecast_series, label='Forecasted Price', color='red', linestyle='--')
plt.scatter([pd.to_datetime(purchase_date_str)], [purchase_price], color='green', s=80, label='Purchase Price', zorder=5)
plt.scatter([today], [latest_price], color='blue', s=80, label='Latest Price', zorder=5)
plt.scatter([future_target_date], [future_price], color='red', s=80, label='Future Predicted Price', zorder=5)
plt.title(f"{ticker} Stock Price Forecast")
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
