# Stock Market Predictor

## 📊 Project Overview

**Stock Market Predictor** is a comprehensive time series forecasting system that predicts future stock prices for NSE (National Stock Exchange) companies using advanced statistical and machine learning models. The project integrates company financial statements, global macroeconomic indicators, and news sentiment to deliver accurate, context-aware price predictions.

---

## 🎯 Objectives

1. **Predict Future Stock Prices:** Use ARIMA and SARIMA models to forecast stock prices for any specified future date.
2. **Integrate Financial Data:** Leverage balance sheets, income statements, cash flows, and annual reports for enhanced predictions.
3. **Incorporate External Factors:** Include global macroeconomic data and sentiment scores to capture real-world market influences.
4. **Calculate Investment Returns:** Display profit/loss scenarios from purchase date to predicted future date.
5. **Provide Interpretability:** Use transparent statistical models (ARIMA/SARIMA) for explainable forecasts.

---

## 🏗️ Technical Architecture

### **Components:**

| Component | Description |
|-----------|-------------|
| **Frontend** | Streamlit-based interactive interface with responsive dashboard |
| **Backend** | Python with FastMCP framework for modular financial tools |
| **AI Layer** | ARIMA/SARIMA models with exogenous financial variables |
| **Data Sources** | yfinance (historical prices, financials), custom APIs for sentiment/macro data |

### **System Workflow:**

```
User Input (Stock Symbol, Dates)
         ↓
Download Price & Financial Data (yfinance)
         ↓
Preprocess & Align Features (by trading date)
         ↓
Model Selection (ARIMA for ≤60 days, SARIMA for longer)
         ↓
Forecast Future Prices
         ↓
Visualize & Calculate Profit/Loss
         ↓
Output Results
```

---

## 🔧 Technologies & Libraries

- **Python 3.x**
- **yfinance** — Download stock prices and financial statements
- **statsmodels** — ARIMA/SARIMA model implementations
- **pmdarima** — Auto-ARIMA parameter selection
- **pandas** — Data manipulation and alignment
- **numpy** — Numerical computations
- **matplotlib** — Visualization of forecasts and historical data
- **scikit-learn** — Data preprocessing and scaling (optional)

---

## 📥 Installation

### **Prerequisites:**
- Python 3.7 or higher
- pip package manager

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/arMOULD560/Stock-Market-Predictor.git
cd Stock-Market-Predictor
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Run the Application**
```bash
python main.py
```

---

## 📋 Requirements

Create a `requirements.txt` file with:

```
yfinance==0.2.32
pandas==2.0.3
numpy==1.24.3
statsmodels==0.14.0
pmdarima==2.0.4
matplotlib==3.7.2
scikit-learn==1.3.0
```

---

## 🚀 Usage

### **Basic Usage:**

1. **Run the script:**
   ```bash
   python main.py
   ```

2. **Enter required inputs:**
   - NSE stock symbol (e.g., `RELIANCE`, `TCS`, `INFY`)
   - Purchase date (YYYY-MM-DD format)
   - Future date for prediction (YYYY-MM-DD format)

3. **View Results:**
   - **Console Output:**
     - Purchase price and date
     - Latest price and date
     - Profit/Loss from purchase to today
     - Predicted future price
     - Projected profit/loss
   
   - **Graph:**
     - Historical price trend (blue line)
     - Forecasted prices (red dashed line)
     - Key markers (purchase, latest, predicted prices)

### **Example:**
```
Enter NSE stock symbol (example: RELIANCE): TCS
Enter Date of Purchase (YYYY-MM-DD): 2023-01-15
Enter Future Date to Predict Price (YYYY-MM-DD): 2025-12-31

--- Price Summary for TCS.NS ---
Purchase Date (Adjusted): 2023-01-16, Price: ₹3450.25
Latest Date: 2025-11-17, Price: ₹3920.50
Profit/Loss (Purchase → Today): ₹470.25
Future Date (Adjusted): 2025-12-31, Predicted Price: ₹4150.75
Profit/Loss (Today → Future): ₹230.25
Profit/Loss (Purchase → Future): ₹700.50
```

---

## 📊 Understanding ARIMA & SARIMA

### **ARIMA (AutoRegressive Integrated Moving Average)**

- **Use Case:** Short-term forecasts (up to 60 business days) without strong seasonality
- **Components:**
  - **AR (p):** Uses past values as predictors
  - **I (d):** Differencing to achieve stationarity
  - **MA (q):** Uses past forecast errors

- **Notation:** ARIMA(p, d, q)

### **SARIMA (Seasonal ARIMA)**

- **Use Case:** Long-term forecasts with seasonal patterns
- **Additional Components:**
  - **Seasonal AR (P), I (D), MA (Q):** Capture repeating cycles
  - **Seasonal Period (m):** 5 for weekly, 20+ for monthly patterns

- **Notation:** SARIMA(p, d, q) × (P, D, Q, m)

### **Model Selection Logic:**

| Forecast Duration | Model | Reason |
|-------------------|-------|--------|
| ≤ 60 business days | ARIMA | Faster, accurate for short-term |
| > 60 business days | SARIMA | Captures seasonal patterns, long-term trends |

---

## 💰 Data Integrated

### **Company Financials:**
- Balance Sheet (Total Assets, Liabilities, Current Liabilities)
- Income Statement (Revenue, Gross Profit, Operating Income, Net Income)
- Cash Flow Statement (Operating, Investing, Financing Cash Flows)
- Annual & Quarterly Reports

### **External Features (Optional, Placeholder):**
- News Sentiment Scores (via NLP/APIs)
- Macroeconomic Indicators (Interest Rates, Inflation, GDP, FX Rates)
- Global Market Indices (S&P 500, Nifty 50, Oil Prices)
- Current Affairs Impact Indicators

---

## 📈 Features

✅ **Automatic Model Selection** — Chooses ARIMA or SARIMA based on forecast horizon  
✅ **Comprehensive Financial Integration** — Uses all available balance sheet, income statement, and cash flow data  
✅ **Data Cleaning & Preprocessing** — Handles missing values, infinities, and outliers  
✅ **Business Day Alignment** — Respects trading calendar (excludes weekends/holidays)  
✅ **Visual Output** — Clear graphs showing price history, forecast, and key milestones  
✅ **Profit/Loss Calculation** — Shows investment returns for the specified period  
✅ **Scalability** — Easy to extend with additional stocks, indicators, or models  

---

## 🔍 Project Structure

```
Stock-Market-Predictor/
├── main.py                          # Main application file
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
├── images/
│   ├── architecture_flow.png       # Technical architecture diagram
│   ├── sample_forecast.png         # Example forecast output
│   └── project_demo.png            # Project demonstration
└── LICENSE                          # Project license (optional)
```

---

## 📊 Sample Output

### **Console Output:**
```
--- Price Summary for RELIANCE.NS ---
Purchase Date (Adjusted): 2022-06-15, Price: ₹2450.00
Latest Date: 2025-11-17, Price: ₹3100.00
Profit/Loss (Purchase → Today): ₹650.00
Future Date (Adjusted): 2025-12-15, Predicted Price: ₹3250.00
Profit/Loss (Today → Future): ₹150.00
Profit/Loss (Purchase → Future): ₹800.00
```

### **Visualization:**
- Historical price trend (blue line)
- Forecasted future prices (red dashed line)
- Purchase price marker (green dot)
- Latest price marker (blue dot)
- Predicted future price marker (red dot)

---

## ⚠️ Limitations

1. **ARIMA/SARIMA Constraints:** Models work best with stationary data and may struggle with abrupt market shocks or structural breaks.
2. **Future Exogenous Variables:** For long-term SARIMAX forecasts, assumes last known financial values extend forward (naive approximation).
3. **Market Unpredictability:** Stock prices are influenced by countless factors; predictions should not be treated as guaranteed investment advice.
4. **Data Quality:** Predictions depend on data quality from yfinance and external sources.

---

## 🚀 Future Enhancements

- Integrate **LSTM/Deep Learning** models for complex nonlinear patterns
- Add **ensemble methods** combining multiple model forecasts
- Include **real-time sentiment analysis** from news APIs and social media
- Implement **confidence intervals** for forecast uncertainty quantification
- Build **web dashboard** with interactive visualizations
- Support **multiple assets** and **portfolio analysis**
- Add **backtesting framework** to validate model performance

---

## 📚 References & Resources

### **Time Series Forecasting:**
- [ARIMA Models Explained](https://www.datacamp.com/blog/arima-time-series-forecasting)
- [SARIMA & Seasonality](https://geeksforgeeks.org/sarima-model/)
- [statsmodels Documentation](https://www.statsmodels.org/)
- [pmdarima: Auto-ARIMA](https://alkaline-ml.com/pmdarima/)

### **Financial Data:**
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [NSE India Stock Exchange](https://www.nseindia.com/)

### **Time Series Best Practices:**
- Box & Jenkins ARIMA Methodology
- Stationarity Testing (ADF, KPSS tests)
- Autocorrelation & Partial Autocorrelation Analysis

---

## 👨‍💻 Contributors

- **Developer:** [Your Name]
- **Project:** Stock Market Predictor (Academic/Professional Project)
- **Last Updated:** November 2025

---

## 📄 License

This project is open-source and available under the MIT License. See `LICENSE` file for details.

---

## 🤝 Support & Feedback

For issues, suggestions, or contributions:
1. Open an **Issue** on GitHub
2. Submit a **Pull Request** with improvements
3. Share feedback or use cases

---

## ⭐ Acknowledgments

- **yfinance** team for easy stock data access
- **statsmodels** and **pmdarima** communities for robust time series tools
- Academic resources and financial data providers

---

**Happy Forecasting! 📈**
