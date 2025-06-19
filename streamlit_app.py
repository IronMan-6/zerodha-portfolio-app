import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

st.set_page_config(page_title="Zerodha Portfolio Dashboard", layout="wide")
st.title("📈 Zerodha Portfolio Analyzer (AI-Enhanced)")

uploaded_file = st.file_uploader("Upload your Zerodha holdings CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # --- Clean Columns ---
    df.columns = df.columns.str.strip().str.replace(".", "", regex=False)
    df.rename(columns={
        "Instrument": "Stock",
        "Qty": "Quantity",
        "Avg cost": "AvgCost",
        "LTP": "CurrentPrice",
        "Invested": "InvestedAmount",
        "Cur val": "CurrentValue",
        "P&L": "PL",
        "Net chg": "NetChangePercent",
        "Day chg": "DayChangePercent"
    }, inplace=True)
    
    df["Return%"] = (df["PL"] / df["InvestedAmount"]) * 100

    # --- Sector Classification ---
    sector_map = {
        "BANKINDIA": "Banking",
        "UNIONBANK": "Banking",
        "INDIANB": "Banking",
        "PNB": "Banking",
        "FEDERALBNK": "Banking",
        "INDUSINDBK": "Banking",
        "GOLDBEES": "Commodities",
        "INFY": "IT Services",
        "DRREDDY": "Pharmaceuticals",
        "NATCOPHARM": "Pharmaceuticals",
        "MANAPPURAM": "Finance",
        "TATAMOTORS": "Automobile",
        "NTPCGREEN": "Energy",
        "BAJAJHFL": "Finance",
        "ARE&M": "Industrials",
        "ITCHOTELS": "Hospitality"
    }

    df["Sector"] = df["Stock"].map(sector_map).fillna("Other")

    # --- Portfolio Summary ---
    st.subheader("📊 Portfolio Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Invested (₹)", f"{df['InvestedAmount'].sum():,.2f}")
    col2.metric("Current Value (₹)", f"{df['CurrentValue'].sum():,.2f}")
    col3.metric("Unrealized P&L (₹)", f"{df['PL'].sum():,.2f}")
    col4.metric("Overall Return (%)", f"{(df['PL'].sum()/df['InvestedAmount'].sum()*100):.2f}%")

    # --- Investment vs Value Chart ---
    st.subheader("💼 Investment vs Current Value")
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.4
    index = np.arange(len(df))
    ax.bar(index, df["InvestedAmount"], bar_width, label="Invested", alpha=0.6)
    ax.bar(index + bar_width, df["CurrentValue"], bar_width, label="Current Value", alpha=0.8)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(df["Stock"], rotation=45, ha="right")
    ax.set_ylabel("₹ Amount")
    ax.legend()
    st.pyplot(fig)

    # --- Return Chart ---
    st.subheader("📈 Stock-wise Return %")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(x="Stock", y="Return%", data=df.sort_values("Return%", ascending=False), palette="coolwarm", ax=ax2)
    ax2.axhline(0, color="black", linestyle="--")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig2)

    # --- Sector Pie ---
    st.subheader("🏷️ Sector Allocation")
    sector_group = df.groupby("Sector")["CurrentValue"].sum().sort_values(ascending=False)
    fig4, ax4 = plt.subplots()
    ax4.pie(sector_group, labels=sector_group.index, autopct="%1.1f%%", startangle=140)
    ax4.axis("equal")
    st.pyplot(fig4)

    # --- AI-Driven Portfolio Tips ---
    st.subheader("💡 AI Analyst Insights")
    top_gainers = df[df["Return%"] > 100]["Stock"].tolist()
    top_laggards = df[df["Return%"] < -20]["Stock"].tolist()
    exposure = df["Sector"].value_counts(normalize=True) * 100

    with st.container():
        if top_gainers:
            st.write(f"📈 **Consider Partial Profit Booking** in: `{', '.join(top_gainers)}`")
        if top_laggards:
            st.write(f"⚠️ **Reevaluate Laggards** like: `{', '.join(top_laggards)}`")
        if "Banking" in exposure and exposure["Banking"] > 40:
            st.write("🧯 **Heavy Banking Allocation** (>40%)—consider diversifying into other sectors.")
        if "IT Services" not in exposure.values:
            st.write("🔍 Consider adding defensive or growth exposure in IT, FMCG, or Large Cap Index ETFs.")
        if "Commodities" in exposure and exposure["Commodities"] < 10:
            st.write("🪙 Gold hedge is present but small—fine for inflation protection.")

    # --- Risk Metrics ---
    st.subheader("📉 Risk Metrics: Beta & Sharpe Ratio")
    try:
        benchmark_data = yf.download("^NSEI", period="1y")
        if "Adj Close" in benchmark_data.columns:
            benchmark = benchmark_data["Adj Close"].pct_change().dropna()
            betas, sharpes = {}, {}
            risk_free = 0.065 / 252

            for stock in df["Stock"]:
                try:
                    s_data = yf.download(stock + ".NS", period="1y")["Adj Close"].pct_change().dropna()
                    combined = pd.concat([s_data, benchmark], axis=1).dropna()
                    combined.columns = ["stock", "benchmark"]
                    beta = np.cov(combined["stock"], combined["benchmark"])[0, 1] / np.var(combined["benchmark"])
                    sharpe = (combined["stock"].mean() - risk_free) / combined["stock"].std()
                    betas[stock] = round(beta, 2)
                    sharpes[stock] = round(sharpe, 2)
                except:
                    betas[stock] = "N/A"
                    sharpes[stock] = "N/A"

            risk_df = pd.DataFrame({
                "Stock": betas.keys(),
                "Beta": betas.values(),
                "Sharpe Ratio": sharpes.values()
            })
            st.dataframe(risk_df)
        else:
            st.warning("⚠️ Benchmark data unavailable—risk metrics skipped.")
    except Exception as e:
        st.error(f"⚠️ Risk metrics couldn't be calculated: {e}")

    # --- Correlation Heatmap ---
    st.subheader("📌 Correlation Heatmap (1Y Returns)")
    price_data = {}
    for stock in df["Stock"]:
        try:
            s = yf.download(stock + ".NS", period="1y")["Adj Close"]
            price_data[stock] = s
        except:
            continue

    if price_data:
        price_df = pd.DataFrame(price_data).pct_change().dropna()
        corr = price_df.corr()
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)
    else:
        st.warning("⚠️ Could not fetch stock data for correlation matrix.")

    st.success("✅ Analysis complete! Upload updated data anytime to refresh.")

else:
    st.info("👆 Upload your Zerodha holdings CSV to begin analysis.")
