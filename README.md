# Zerodha Portfolio Analyzer 📈

This is a powerful, multi-tab Streamlit dashboard that transforms your Zerodha holdings CSV into visual insights, strategy simulations, and performance comparisons—backed by AI logic.

## 🚀 Features

🔹 **Dashboard**  
- Upload your `holdings.csv` directly from Zerodha Console  
- Visualize:
  - Stock-wise performance
  - Sector diversification
  - Beta & Sharpe risk metrics
  - Return heatmaps and correlations  
- Built-in AI insights: laggard alerts, overexposure warnings, profit-taking suggestions

🔹 **Simulator**  
- Choose a stock to "sell" and reallocate to others  
- Simulate sector rebalancing effects  
- Visual feedback via allocation pie chart

🔹 **Compare Portfolios**  
- Upload two CSVs (before/after rebalancing)  
- Automatically detect:
  - Qty, value & P&L shifts
  - Stock additions/removals  
- Visualize changes with comparison bar charts

## 🧠 Tech Stack

- Python, Streamlit  
- Pandas, NumPy, Matplotlib, Seaborn  
- Yahoo Finance API via `yfinance`

## 💾 How to Use

1. Export `holdings.csv` from [Zerodha Console](https://console.zerodha.com/portfolio/holdings)
2. Launch the app at:  
   👉 [**Your Deployed Streamlit Link**](https://your-streamlit-app-url)
3. Use the tabs to switch between:
   - Real-time Dashboard
   - Reallocation Simulator
   - Portfolio Comparison

## 📦 Installation (For Local Use)

```bash
git clone https://github.com/yourusername/zerodha-portfolio-app.git
cd zerodha-portfolio-app
pip install -r requirements.txt
streamlit run streamlit_app.py
