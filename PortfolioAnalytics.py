import streamlit as st
import pandas as pd
import tempfile
import os
import subprocess
import sys
from datetime import date, datetime
import requests

# --- AUTO-INSTALL DEPENDENCIES ---
def install_missing_packages():
    """Install missing packages automatically"""
    required_packages = [
        "yfinance==0.2.18",
        "quantstats==0.0.62", 
        "plotly==5.15.0",
        "pandas==2.0.3",
        "requests==2.31.0"
    ]
    
    for package in required_packages:
        try:
            if "yfinance" in package:
                import yfinance
            elif "quantstats" in package:
                import quantstats
            elif "plotly" in package:
                import plotly
        except ImportError:
            st.warning(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                st.success(f"✅ {package} installed successfully!")
            except Exception as e:
                st.error(f"Failed to install {package}: {str(e)}")
                return False
    return True

# Install dependencies first
if install_missing_packages():
    st.success("✅ All dependencies installed successfully!")
else:
    st.error("❌ Some dependencies failed to install. The app may not work properly.")

# Now import the packages
try:
    import yfinance as yf
    import quantstats as qs
    import plotly.express as px
    st.success("✅ All packages imported successfully!")
except ImportError as e:
    st.error(f"❌ Failed to import packages: {str(e)}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="NSE Portfolio Analytics Dashboard", 
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}

if 'analysis_generated' not in st.session_state:
    st.session_state.analysis_generated = False

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def load_stock_list():
    """Load complete NSE stock list"""
    try:
        url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        df = pd.read_csv(url)
        df.rename(columns={"SYMBOL": "Symbol", "NAME OF COMPANY": "Company"}, inplace=True)
        return df[["Symbol", "Company"]]
    except Exception as e:
        st.error(f"Error loading stock list: {str(e)}")
        return pd.DataFrame()

def get_live_price(symbol):
    """Get live price for a stock"""
    try:
        ticker = f"{symbol}.NS"
        stock = yf.Ticker(ticker)
        
        # Get price data using history (most reliable method)
        hist = stock.history(period="2d")  # Get 2 days to calculate change
        if len(hist) >= 2:
            current_price = hist['Close'].iloc[-1]
            previous_close = hist['Close'].iloc[-2]
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100
            company_name = stock.info.get('longName', symbol)
            return current_price, change, change_percent, company_name
        elif len(hist) == 1:
            # If only one day available
            current_price = hist['Close'].iloc[0]
            return current_price, 0, 0, symbol
        else:
            return None, None, None, None
    except Exception as e:
        return None, None, None, None

def search_stocks(query, stocks_df):
    """Search stocks by symbol or company name"""
    if not query or len(query) < 2:
        return pd.DataFrame()
    
    query = query.upper()
    mask = (stocks_df['Symbol'].str.upper().str.contains(query)) | \
           (stocks_df['Company'].str.upper().str.contains(query))
    
    return stocks_df[mask].head(15)

def add_to_portfolio(symbol, weight, company_name):
    """Add stock to portfolio"""
    if symbol in st.session_state.portfolio:
        st.warning(f"{symbol} is already in your portfolio!")
        return False
    
    st.session_state.portfolio[symbol] = {
        'weight': weight,
        'company': company_name,
        'ticker': f"{symbol}.NS"
    }
    st.session_state.analysis_generated = False
    return True

def remove_from_portfolio(symbol):
    """Remove stock from portfolio"""
    if symbol in st.session_state.portfolio:
        del st.session_state.portfolio[symbol]
        st.session_state.analysis_generated = False
        return True
    return False

def update_portfolio_weight(symbol, new_weight):
    """Update weight for a stock in portfolio"""
    if symbol in st.session_state.portfolio:
        st.session_state.portfolio[symbol]['weight'] = new_weight
        st.session_state.analysis_generated = False
        return True
    return False

def normalize_portfolio_weights():
    """Normalize weights to sum to 100%"""
    total_weight = sum(stock['weight'] for stock in st.session_state.portfolio.values())
    if total_weight > 0:
        for symbol in st.session_state.portfolio:
            st.session_state.portfolio[symbol]['weight'] = (st.session_state.portfolio[symbol]['weight'] / total_weight) * 100
        st.session_state.analysis_generated = False
        return True
    return False

@st.cache_data(ttl=600)
def download_portfolio_data(tickers, start_date, end_date):
    """Download historical data with caching"""
    try:
        # Convert to string for yfinance
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        data = yf.download(tickers, start=start_str, end=end_str, progress=False)["Adj Close"]
        return data
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return pd.DataFrame()

def calculate_portfolio_analysis(start_date, end_date):
    """Calculate portfolio analysis with validation"""
    if not st.session_state.portfolio:
        return None, None, "Portfolio is empty"
    
    try:
        if start_date >= end_date:
            return None, None, "Start date must be before end date"
        
        tickers = [stock['ticker'] for stock in st.session_state.portfolio.values()]
        weights = [stock['weight'] for stock in st.session_state.portfolio.values()]
        
        total_weight = sum(weights)
        if total_weight <= 0:
            return None, None, "Total portfolio weight must be greater than 0"
        
        weights = [w / total_weight for w in weights]
        
        data = download_portfolio_data(tickers, start_date, end_date)
        
        if data.empty:
            return None, None, "No historical data available for the selected stocks and period"
        
        returns = data.pct_change().dropna()
        
        if returns.empty:
            return None, None, "No return data available after processing"
        
        portfolio_returns = (returns * weights).sum(axis=1)
        
        if len(portfolio_returns) < 5:
            return None, None, "Insufficient data points for analysis"
        
        if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
            portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
        
        return portfolio_returns, data, "Success"
        
    except Exception as e:
        return None, None, f"Error in portfolio analysis: {str(e)}"

# --- Main Application UI ---
st.title("📊 NSE Portfolio Analytics Dashboard")
st.markdown("Search stocks, build your portfolio, and analyze performance")

# Sidebar for search and portfolio management
with st.sidebar:
    st.header("🔍 Stock Search")
    
    # Load stock list
    stocks_df = load_stock_list()
    
    if not stocks_df.empty:
        st.success(f"✅ {len(stocks_df)} stocks loaded")
    
    # Search input
    query = st.text_input("Search by symbol or company name:", placeholder="e.g., RELIANCE, TCS...")
    
    # Search results
    if query and len(query) >= 2:
        results = search_stocks(query, stocks_df)
        if not results.empty:
            st.write(f"**Found {len(results)} stocks:**")
            for _, row in results.iterrows():
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"**{row['Symbol']}**")
                    st.write(f"_{row['Company']}_")
                with col2:
                    price, change, pct, _ = get_live_price(row['Symbol'])
                    if price:
                        st.write(f"₹{price:.1f}")
                with col3:
                    if st.button("Add", key=f"add_{row['Symbol']}"):
                        add_to_portfolio(row['Symbol'], 0, row['Company'])
                        st.rerun()
                st.divider()
    
    # Portfolio management
    st.header("📊 Your Portfolio")
    
    if st.session_state.portfolio:
        total_weight = sum(stock['weight'] for stock in st.session_state.portfolio.values())
        
        st.metric("Total Stocks", len(st.session_state.portfolio))
        st.metric("Total Weight", f"{total_weight:.1f}%")
        
        for symbol, data in st.session_state.portfolio.items():
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"**{symbol}**")
            with col2:
                new_weight = st.number_input(
                    "Weight %",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(data['weight']),
                    key=f"weight_{symbol}",
                    step=1.0
                )
                if new_weight != data['weight']:
                    update_portfolio_weight(symbol, new_weight)
            with col3:
                if st.button("❌", key=f"del_{symbol}"):
                    remove_from_portfolio(symbol)
                    st.rerun()
            st.divider()
        
        if abs(total_weight - 100.0) > 0.1:
            st.warning(f"Weights sum to {total_weight:.1f}% (should be 100%)")
            if st.button("🔄 Normalize Weights"):
                normalize_portfolio_weights()
                st.rerun()
        
        if st.button("🗑️ Clear Portfolio"):
            st.session_state.portfolio = {}
            st.session_state.analysis_generated = False
            st.rerun()
    else:
        st.info("Search and add stocks to build your portfolio")

# Main analysis area
st.header("📈 Portfolio Analysis")

# Date selection
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=date(2023, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=date.today())

# Analysis button
if st.button("🚀 Generate Portfolio Analysis", type="primary", use_container_width=True):
    if not st.session_state.portfolio:
        st.error("Please add stocks to your portfolio first!")
    else:
        total_weight = sum(stock['weight'] for stock in st.session_state.portfolio.values())
        if abs(total_weight - 100.0) > 0.1:
            st.error(f"Portfolio weights must sum to 100% (current: {total_weight:.1f}%)")
        elif start_date >= end_date:
            st.error("Start date must be before end date")
        else:
            st.session_state.analysis_generated = True
            st.rerun()

# Display analysis results
if st.session_state.analysis_generated and st.session_state.portfolio:
    with st.spinner("Analyzing portfolio..."):
        portfolio_returns, stock_data, msg = calculate_portfolio_analysis(start_date, end_date)
        
        if portfolio_returns is not None:
            st.success("Analysis Complete!")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                sharpe = qs.stats.sharpe(portfolio_returns)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            with col2:
                max_dd = qs.stats.max_drawdown(portfolio_returns) * 100
                st.metric("Max Drawdown", f"{max_dd:.2f}%")
            with col3:
                cagr = qs.stats.cagr(portfolio_returns) * 100
                st.metric("CAGR", f"{cagr:.2f}%")
            with col4:
                vol = qs.stats.volatility(portfolio_returns) * 100
                st.metric("Volatility", f"{vol:.2f}%")
            
            # Charts
            col1, col2 = st.columns(2)
            with col1:
                # Allocation pie chart
                allocation_data = {
                    'Symbol': list(st.session_state.portfolio.keys()),
                    'Weight': [stock['weight'] for stock in st.session_state.portfolio.values()]
                }
                fig_pie = px.pie(allocation_data, values='Weight', names='Symbol', title="Portfolio Allocation")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Cumulative returns
                cumulative_returns = (1 + portfolio_returns).cumprod()
                fig_line = px.line(x=cumulative_returns.index, y=cumulative_returns.values, 
                                 title="Portfolio Growth", labels={'x': 'Date', 'y': 'Value'})
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.error(f"Analysis failed: {msg}")

st.markdown("---")
st.markdown("Built with Streamlit • Data from NSE India & Yahoo Finance")



