import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import datetime

st.set_page_config(page_title="Zerodha Portfolio Analyzer", layout="wide")
st.title("📈 Zerodha Portfolio Dashboard")

# --- Upload Holdings CSV ---
uploaded_file = st.file_uploader("📂 Upload your Zerodha holdings CSV", type="csv")
data_loaded = uploaded_file is not None

# --- Mapping ---
sector_map = {
    "BANKINDIA": "Banking", "UNIONBANK": "Banking", "INDIANB": "Banking", "PNB": "Banking",
    "FEDERALBNK": "Banking", "INDUSINDBK": "Banking", "GOLDBEES": "Commodities",
    "INFY": "IT Services", "DRREDDY": "Pharmaceuticals", "NATCOPHARM": "Pharmaceuticals",
    "MANAPPURAM": "Finance", "TATAMOTORS": "Automobile", "NTPCGREEN": "Energy",
    "BAJAJHFL": "Finance", "ITCHOTELS": "Hospitality"
}

symbol_map = {
    "BAJAJHFL": "BAJFINANCE.NS", "BANKINDIA": "BANKINDIA.NS", "DRREDDY": "DRREDDY.NS",
    "FEDERALBNK": "FEDERALBNK.NS", "GOLDBEES": "GOLDBEES.NS", "INDIANB": "INDIANB.NS",
    "INDUSINDBK": "INDUSINDBK.NS", "INFY": "INFY.NS", "MANAPPURAM": "MANAPPURAM.NS",
    "NATCOPHARM": "NATCOPHARM.NS", "NTPCGREEN": "NTPC.NS", "PNB": "PNB.NS",
    "TATAMOTORS": "TATAMOTORS.NS", "UNIONBANK": "UNIONBANK.NS", "ITCHOTELS": "ITCHOTEL.NS"
}

nifty50_benchmark = {
    "Banking": 25, "IT Services": 14, "Automobile": 8, "Finance": 7,
    "Pharmaceuticals": 5, "Energy": 12, "Commodities": 6, "Hospitality": 2, "Other": 21
}

sector_caps = {
    "Banking": 30,
    "Pharmaceuticals": 15,
    "Commodities": 10,
    "Energy": 15,
    "Finance": 20
}

# --- Data Prep ---
if data_loaded:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.replace(".", "", regex=False)
    df.rename(columns={
        "Instrument": "Stock", "Qty": "Quantity", "Avg cost": "AvgCost",
        "LTP": "CurrentPrice", "Invested": "InvestedAmount", "Cur val": "CurrentValue",
        "P&L": "PL"
    }, inplace=True, errors="ignore")
    df["Return%"] = (df["PL"] / df["InvestedAmount"]) * 100
    df["Sector"] = df["Stock"].map(sector_map).fillna("Other")

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "🎯 Simulator", "🔍 Compare", "📡 Live Ticker"])

# --- Dashboard Tab ---
with tab1:
    if not data_loaded:
        st.info("Upload a holdings CSV to view dashboard.")
    else:
        st.subheader("💼 Portfolio Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Invested ₹", f"{df['InvestedAmount'].sum():,.0f}")
        col2.metric("Current ₹", f"{df['CurrentValue'].sum():,.0f}")
        col3.metric("P&L ₹", f"{df['PL'].sum():,.0f}")
        col4.metric("Return %", f"{(df['PL'].sum()/df['InvestedAmount'].sum()*100):.2f}%")

        st.subheader("📊 Value Comparison")
        fig, ax = plt.subplots()
        index = np.arange(len(df))
        ax.bar(index, df["InvestedAmount"], 0.4, label="Invested", alpha=0.6)
        ax.bar(index + 0.4, df["CurrentValue"], 0.4, label="Current", alpha=0.8)
        ax.set_xticks(index + 0.2)
        ax.set_xticklabels(df["Stock"], rotation=45, ha="right")
        ax.set_ylabel("₹")
        ax.legend()
        st.pyplot(fig)

        st.subheader("📈 Stock-wise Return %")
        fig2, ax2 = plt.subplots()
        sns.barplot(x="Stock", y="Return%", data=df, palette="coolwarm", ax=ax2)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        st.subheader("🏷️ Sector Allocation")
        sector_group = df.groupby("Sector")["CurrentValue"].sum()
        fig3, ax3 = plt.subplots()
        ax3.pie(sector_group, labels=sector_group.index, autopct="%1.1f%%", startangle=140)
        ax3.axis("equal")
        st.pyplot(fig3)

        st.subheader("💡 AI Portfolio Insights")
        if any(df["Return%"] > 100):
            st.success("📈 Consider booking profits in: " + ", ".join(df[df["Return%"] > 100]["Stock"]))
        if any(df["Return%"] < -20):
            st.warning("🔻 Laggards to review: " + ", ".join(df[df["Return%"] < -20]["Stock"]))
        # --- Risk Metrics ---
        st.subheader("📉 Risk Metrics")
        try:
            benchmark = yf.download("^NSEI", period="1y")["Adj Close"].pct_change().dropna()
            data = []
            for stock in df["Stock"]:
                ticker = symbol_map.get(stock, stock + ".NS")
                try:
                    s = yf.download(ticker, period="1y")["Adj Close"].pct_change().dropna()
                    aligned = pd.concat([s, benchmark], axis=1).dropna()
                    aligned.columns = ["stock", "benchmark"]
                    beta = np.cov(aligned.T)[0, 1] / np.var(aligned["benchmark"])
                    sharpe = (aligned["stock"].mean() - 0.065/252) / aligned["stock"].std()
                    data.append((stock, round(beta, 2), round(sharpe, 2)))
                except:
                    continue
            if data:
                st.dataframe(pd.DataFrame(data, columns=["Stock", "Beta", "Sharpe Ratio"]))
            else:
                st.warning("⚠️ No valid risk data.")
        except:
            st.warning("⚠️ Benchmark data unavailable.")

        # --- Correlation Heatmap ---
        st.subheader("📌 Correlation Matrix")
        price_data = {}
        for stock in df["Stock"]:
            ticker = symbol_map.get(stock, stock + ".NS")
            try:
                s = yf.download(ticker, period="1y")["Adj Close"]
                price_data[stock] = s
            except:
                continue
        valid_prices = {k: v for k, v in price_data.items() if isinstance(v, pd.Series) and not v.empty}
        if valid_prices:
            df_corr = pd.DataFrame(valid_prices).pct_change().dropna()
            figc, axc = plt.subplots(figsize=(8, 6))
            sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", ax=axc)
            st.pyplot(figc)
        else:
            st.warning("⚠️ Not enough data for correlation.")

        # --- Sector Rebalancing: Benchmark + Max Caps ---
        st.subheader("🧭 Rebalancing Suggestions (vs Benchmark + Max Rules)")
        current = df.groupby("Sector")["CurrentValue"].sum()
        total = current.sum()
        current_weights = (current / total * 100).round(2)
        benchmark = pd.Series(nifty50_benchmark).reindex(current_weights.index).fillna(0)
        max_caps = pd.Series(sector_caps).reindex(current_weights.index).fillna(100)

        rebal_df = pd.DataFrame({
            "Current %": current_weights,
            "Nifty50 %": benchmark,
            "Max Allowed %": max_caps,
            "Delta to Nifty": (benchmark - current_weights).round(2)
        })

        st.dataframe(rebal_df)

        for sector, row in rebal_df.iterrows():
            if row["Current %"] > row["Max Allowed %"]:
                st.warning(f"⚠️ {sector} exceeds max allowed ({row['Current %']}% > {row['Max Allowed %']}%)")
            elif row["Delta to Nifty"] > 3:
                st.info(f"📥 Increase allocation to {sector} by ~{row['Delta to Nifty']}%")
            elif row["Delta to Nifty"] < -3:
                st.info(f"📤 Consider trimming {sector} by ~{abs(row['Delta to Nifty'])}%")

# --- Simulator Tab ---
with tab2:
    st.header("🎯 Scenario Simulator")
    if not data_loaded:
        st.info("Upload your holdings to simulate.")
    else:
        stock_to_sell = st.selectbox("Stock to Sell", df["Stock"].unique())
        amount = st.number_input("Reallocate ₹", min_value=100.0, value=1000.0)
        targets = st.multiselect("Stocks to Buy", df["Stock"].unique())
        if st.button("Simulate") and targets:
            per_stock = amount / len(targets)
            alloc_df = pd.DataFrame({"₹ Allocated": [per_stock] * len(targets)}, index=targets)
            alloc_df["Sector"] = alloc_df.index.map(df.set_index("Stock")["Sector"].to_dict())
            st.dataframe(alloc_df)
            figsim, axsim = plt.subplots()
            pie_data = alloc_df.groupby("Sector")["₹ Allocated"].sum()
            axsim.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%")
            axsim.axis("equal")
            st.pyplot(figsim)

# --- Compare Tab ---
with tab3:
    st.header("🔍 Compare Two Portfolios")
    c1, c2 = st.columns(2)
    file_old = c1.file_uploader("📁 Older holdings.csv", type="csv", key="old")
    file_new = c2.file_uploader("📁 Newer holdings.csv", type="csv", key="new")
    if file_old and file_new:
        df_old = pd.read_csv(file_old)
        df_new = pd.read_csv(file_new)
        for d in (df_old, df_new):
            d.columns = d.columns.str.strip().str.replace(".", "", regex=False)
            d.rename(columns={"Instrument": "Stock", "Cur val": "CurrentValue", "Qty": "Quantity", "P&L": "PL"}, inplace=True)
        merged = pd.merge(df_old, df_new, on="Stock", suffixes=("_old", "_new"), how="outer")
        merged["Qty Δ"] = merged["Quantity_new"].fillna(0) - merged["Quantity_old"].fillna(0)
        merged["Val Δ"] = merged["CurrentValue_new"].fillna(0) - merged["CurrentValue_old"].fillna(0)
        merged["P&L Δ"] = merged["PL_new"].fillna(0) - merged["PL_old"].fillna(0)
        st.subheader("📋 Comparison Table")
        st.dataframe(merged[["Stock", "Qty Δ", "Val Δ", "P&L Δ"]].sort_values("Val Δ", ascending=False))
        figcomp, axcomp = plt.subplots(figsize=(10, 4))
        sns.barplot(x="Stock", y="Val Δ", data=merged.sort_values("Val Δ", ascending=False), palette="viridis", ax=axcomp)
        axcomp.set_title("Change in Portfolio Value")
        plt.xticks(rotation=45)
        st.pyplot(figcomp)

# --- Live Ticker Tab ---
with tab4:
    st.header("📡 Real-Time Price Pulse")
    if not data_loaded:
        st.info("Upload your holdings to track live prices.")
    else:
        tickers = [symbol_map.get(sym, sym + ".NS") for sym in df["Stock"].unique()]
        st.caption("Tracked Stocks: " + ", ".join(tickers))

        if st.button("🔄 Refresh Prices"):
            latest_data = []
            for tick in tickers:
                try:
                    info = yf.Ticker(tick).info
                    price = info.get("regularMarketPrice")
                    prev_close = info.get("regularMarketPreviousClose")
                    change = price - prev_close if price and prev_close else 0
                    pct = (change / prev_close * 100) if prev_close else 0
                    ts = info.get("regularMarketTime")
                    ts_fmt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "N/A"
                    latest_data.append({
                        "Ticker": tick,
                        "Price (₹)": round(price, 2) if price else None,
                        "Δ %": round(pct, 2),
                        "Last Updated": ts_fmt
                    })
                except:
                    latest_data.append({
                        "Ticker": tick,
                        "Price (₹)": None,
                        "Δ %": None,
                        "Last Updated": "⚠️ Error"
                    })

            df_live = pd.DataFrame(latest_data)
            def highlight_row(val):
                color = "#d4f4dd" if val["Δ %"] > 0 else "#fddddd" if val["Δ %"] < 0 else "white"
                return [f"background-color: {color}"] * len(val)

            st.dataframe(df_live.style.apply(highlight_row, axis=1), use_container_width=True)
        else:
            st.info("Click 'Refresh Prices' to load live data.")
