import streamlit as st
import yfinance as yf
import quantstats as qs
import pandas as pd
import plotly.express as px
import tempfile

st.set_page_config(page_title="Portfolio Analytics Dashboard", layout="wide")
st.title("📊 Portfolio Analytics Dashboard using Quantstats")

st.sidebar.header("⚙️ Portfolio Configuration")

nse_tickers = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDIGO.NS", "INFY.NS", "ITC.NS",
    "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "MAXHEALTH.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS"
]

tickers = st.sidebar.multiselect(
    "Select NSE stocks:",
    options=nse_tickers,
    default=["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS"]
)

if tickers:
    st.sidebar.markdown("### Assign Portfolio Weights")
    weights = []
    for t in tickers:
        w = st.sidebar.slider(
            f"Weight for {t}",
            min_value=0.0,
            max_value=1.0,
            value=round(1 / len(tickers), 2),
            step=0.01
        )
        weights.append(w)
    total = sum(weights)
    if total:
        weights = [w / total for w in weights]

start_date, end_date = st.sidebar.date_input(
    "Select date range:",
    value=(
        pd.to_datetime("2020-01-01"),
        pd.to_datetime("today")
    )
)

generate_btn = st.sidebar.button("🚀 Generate Analysis")

# ===== Main Logic =====
if generate_btn:
    if not tickers:
        st.error("⚠️ Please select at least one stock.")
        st.stop()
    if len(tickers) != len(weights):
        st.error("⚠️ Tickers and weights mismatch.")
        st.stop()

    with st.spinner("Fetching data and computing analytics..."):
        price_data = yf.download(tickers, start=start_date, end=end_date)["Close"]
        returns = price_data.pct_change().dropna()
        portfolio_returns = (returns * weights).sum(axis=1)
        qs.extend_pandas()

    # Display Metrics
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sharpe Ratio", f"{qs.stats.sharpe(portfolio_returns):.2f}")
    with col2:
        st.metric("Max Drawdown", f"{qs.stats.max_drawdown(portfolio_returns) * 100:.2f}%")
    with col3:
        st.metric("CAGR", f"{qs.stats.cagr(portfolio_returns) * 100:.2f}%")
    with col4:
        st.metric("Volatility", f"{qs.stats.volatility(portfolio_returns) * 100:.2f}%")

    # Pie Chart for Weights
    fig_pie = px.pie(
        values=weights,
        names=tickers,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Portfolio Allocation"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Monthly Returns Heatmap
    st.subheader("📈 Monthly Returns Heatmap")
    st.dataframe(portfolio_returns.monthly_returns().style.format("{:.2%}"))

    # Cumulative Returns
    st.subheader("📈 Cumulative Returns")
    st.line_chart((1 + portfolio_returns).cumprod())

    # End-of-Year Returns
    st.subheader("📊 End-of-Year (EOY) Returns")
    eoy_returns = portfolio_returns.resample("Y").apply(lambda x: (x + 1).prod() - 1)
    st.bar_chart(eoy_returns)

    # Generate HTML Report
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
        qs.reports.html(
            portfolio_returns,
            output=tmpfile.name,
            title="Portfolio Performance Report"
        )
    with open(tmpfile.name, "r", encoding="utf-8") as f:
        html_content = f.read()

    st.download_button(
        label="📥 Download Full QuantStats Report",
        data=html_content,
        file_name="portfolio_report.html",
        mime="text/html"
    )

    st.success("✅ Analysis complete! Explore your portfolio metrics above.")




