import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

st.set_page_config(page_title="Zerodha Portfolio Dashboard", layout="wide")
st.title("📈 Zerodha Portfolio Analyzer (AI-Enhanced)")

# --- Tabs ---
tabs = st.tabs(["🏠 Dashboard", "🎯 Simulator", "🔍 Compare"])

# --- Yahoo Symbol Map ---
symbol_map = {
    "BAJAJHFL": "BAJFINANCE.NS",  # example fix
    "BANKINDIA": "BANKINDIA.NS",
    "DRREDDY": "DRREDDY.NS",
    "FEDERALBNK": "FEDERALBNK.NS",
    "GOLDBEES": "GOLDBEES.NS",
    "INDIANB": "INDIANB.NS",
    "INDUSINDBK": "INDUSINDBK.NS",
    "INFY": "INFY.NS",
    "MANAPPURAM": "MANAPPURAM.NS",
    "NATCOPHARM": "NATCOPHARM.NS",
    "NTPCGREEN": "NTPC.NS",
    "PNB": "PNB.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "UNIONBANK": "UNIONBANK.NS"
    # Add more verified mappings here
}

sector_map = {
    "BANKINDIA": "Banking", "UNIONBANK": "Banking", "INDIANB": "Banking", "PNB": "Banking",
    "FEDERALBNK": "Banking", "INDUSINDBK": "Banking", "GOLDBEES": "Commodities",
    "INFY": "IT Services", "DRREDDY": "Pharmaceuticals", "NATCOPHARM": "Pharmaceuticals",
    "MANAPPURAM": "Finance", "TATAMOTORS": "Automobile", "NTPCGREEN": "Energy",
    "BAJAJHFL": "Finance", "ITCHOTELS": "Hospitality"
}

# --- DASHBOARD TAB ---
with tabs[0]:
    file = st.file_uploader("Upload your Zerodha holdings CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.replace(".", "", regex=False)
        df.rename(columns={
            "Instrument": "Stock", "Qty": "Quantity", "Avg cost": "AvgCost",
            "LTP": "CurrentPrice", "Invested": "InvestedAmount", "Cur val": "CurrentValue",
            "P&L": "PL"
        }, inplace=True)
        df["Return%"] = (df["PL"] / df["InvestedAmount"]) * 100
        df["Sector"] = df["Stock"].map(sector_map).fillna("Other")

        st.subheader("📊 Portfolio Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Invested ₹", f"{df['InvestedAmount'].sum():,.0f}")
        col2.metric("Current ₹", f"{df['CurrentValue'].sum():,.0f}")
        col3.metric("P&L ₹", f"{df['PL'].sum():,.0f}")
        col4.metric("Return %", f"{(df['PL'].sum()/df['InvestedAmount'].sum()*100):.2f}%")

        st.subheader("💼 Value Comparison")
        fig, ax = plt.subplots()
        ax.bar(df["Stock"], df["InvestedAmount"], label="Invested", alpha=0.6)
        ax.bar(df["Stock"], df["CurrentValue"], label="Current", alpha=0.8)
        ax.set_xticklabels(df["Stock"], rotation=45, ha="right")
        ax.legend()
        st.pyplot(fig)

        st.subheader("📈 Stock Return %")
        fig2, ax2 = plt.subplots()
        sns.barplot(x="Stock", y="Return%", data=df, palette="coolwarm", ax=ax2)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        st.subheader("🏷️ Sector Allocation")
        fig3, ax3 = plt.subplots()
        sector_group = df.groupby("Sector")["CurrentValue"].sum()
        ax3.pie(sector_group, labels=sector_group.index, autopct="%1.1f%%")
        ax3.axis("equal")
        st.pyplot(fig3)

        # AI Suggestions
        st.subheader("💡 Portfolio Insights")
        if any(df["Return%"] > 100):
            st.write(f"📈 Profit booking ideas: {', '.join(df[df['Return%'] > 100]['Stock'])}")
        if any(df["Return%"] < -20):
            st.write(f"⚠️ Underperformers: {', '.join(df[df['Return%'] < -20]['Stock'])}")

        # RISK Metrics
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
                    pass
            if data:
                st.dataframe(pd.DataFrame(data, columns=["Stock", "Beta", "Sharpe Ratio"]))
            else:
                st.warning("⚠️ Benchmark data unavailable or tickers failed.")
        except:
            st.warning("⚠️ Unable to load benchmark.")

        # CORRELATION
        st.subheader("📌 Correlation Heatmap")
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
            st.warning("⚠️ No valid price data for correlation.")

# SIMULATOR Tab
with tabs[1]:
    st.header("🎯 Scenario Simulator")
    sim_file = st.file_uploader("Upload your holdings CSV for simulation", type="csv", key="sim")
    if sim_file:
        sim_df = pd.read_csv(sim_file)
        if "Cur val" in sim_df.columns:
            sim_df["CurrentValue"] = pd.to_numeric(sim_df["Cur val"], errors="coerce")
        elif "CurrentValue" in sim_df.columns:
            sim_df["CurrentValue"] = pd.to_numeric(sim_df["CurrentValue"], errors="coerce")
        else:
            st.error("❌ Could not find a 'Cur val' or 'CurrentValue' column.")
            st.stop()
        sim_df["Stock"] = sim_df["Instrument"] if "Instrument" in sim_df.columns else sim_df["Stock"]
        stock_to_sell = st.selectbox("Select stock to sell", sim_df["Stock"].unique())
        amount = st.number_input("Amount to simulate reallocation (₹)", min_value=100.0, value=1000.0)
        new_buys = st.multiselect("Buy these stocks hypothetically", sim_df["Stock"].unique())
        if st.button("Simulate"):
            allocation = {s: amount / len(new_buys) for s in new_buys}
            st.write("🔁 Hypothetical Allocation", pd.DataFrame.from_dict(allocation, orient="index", columns=["₹"]))

# COMPARE Tab
with tabs[2]:
    st.header("🔍 Compare Portfolios")
    file1 = st.file_uploader("Older holdings.csv", type="csv", key="old")
    file2 = st.file_uploader("Newer holdings.csv", type="csv", key="new")
    if file1 and file2:
        d1 = pd.read
