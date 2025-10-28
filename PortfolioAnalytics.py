import streamlit as st
import pandas as pd
import tempfile
import os
from datetime import date, datetime
import requests

# --- Dependency Check and Installation ---
def check_and_install_dependencies():
    """Check if required packages are installed, provide guidance if not"""
    missing_packages = []
    
    try:
        import yfinance
    except ImportError:
        missing_packages.append("yfinance")
    
    try:
        import quantstats
    except ImportError:
        missing_packages.append("quantstats")
    
    try:
        import plotly
    except ImportError:
        missing_packages.append("plotly")
    
    if missing_packages:
        st.error(f"❌ Missing required packages: {', '.join(missing_packages)}")
        st.info("""
        **To fix this issue:**
        
        1. **For Streamlit Cloud:** Make sure you have a `requirements.txt` file in your repository with all dependencies
        2. **For Local Run:** Run: `pip install yfinance quantstats plotly pandas requests`
        
        Required packages:
        - streamlit
        - yfinance
        - quantstats  
        - pandas
        - plotly
        - requests
        """)
        return False
    return True

# Check dependencies first
if not check_and_install_dependencies():
    st.stop()

# Now import the packages (they should be available)
import yfinance as yf
import quantstats as qs
import plotly.express as px

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
        st.success(f"✅ Successfully loaded {len(df)} NSE stocks")
        return df[["Symbol", "Company"]]
    except Exception as e:
        st.error(f"❌ Error loading stock list: {str(e)}")
        st.info("Using limited stock list as fallback...")
        # Minimal fallback - only essential stocks
        fallback_stocks = {
            'Symbol': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK'],
            'Company': ['Reliance Industries', 'Tata Consultancy Services', 'HDFC Bank', 'Infosys', 'ICICI Bank']
        }
        return pd.DataFrame(fallback_stocks)

def get_live_price_safe(symbol):
    """Get live price with comprehensive error handling"""
    try:
        ticker = f"{symbol}.NS"
        stock = yf.Ticker(ticker)
        
        # Try multiple methods to get price data
        price_data = None
        
        # Method 1: Try fast_info if available
        if hasattr(stock, 'fast_info'):
            price_data = stock.fast_info
            last_price = getattr(price_data, 'lastPrice', None)
            previous_close = getattr(price_data, 'previousClose', None)
        else:
            # Method 2: Use regular info
            info = stock.info
            last_price = info.get('currentPrice') or info.get('regularMarketPrice')
            previous_close = info.get('previousClose')
        
        # Method 3: Try history as last resort
        if not last_price:
            hist = stock.history(period="1d")
            if not hist.empty:
                last_price = hist['Close'].iloc[-1]
                previous_close = hist['Close'].iloc[0] if len(hist) > 1 else last_price
        
        if last_price and previous_close:
            change = last_price - previous_close
            change_percent = (change / previous_close) * 100
            company_name = stock.info.get('longName', symbol)
            return last_price, change, change_percent, company_name
        
        return None, None, None, None
        
    except Exception as e:
        return None, None, None, None

def search_stocks(query, stocks_df):
    """Search stocks by symbol or company name"""
    if not query or len(query) < 2:
        return pd.DataFrame()
    
    query = query.upper()
    try:
        mask = (stocks_df['Symbol'].str.upper().str.contains(query)) | \
               (stocks_df['Company'].str.upper().str.contains(query))
        return stocks_df[mask].head(15)
    except:
        return pd.DataFrame()

def add_to_portfolio(symbol, weight, company_name):
    """Add stock to portfolio"""
    if symbol in st.session_state.portfolio:
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
def download_portfolio_data_safe(tickers, start_date, end_date):
    """Download historical data with comprehensive error handling"""
    try:
        # Convert dates to string for yfinance
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        data = yf.download(tickers, start=start_str, end=end_str, progress=False, timeout=30)
        
        if data.empty:
            return pd.DataFrame()
            
        # Handle single stock case (different structure)
        if len(tickers) == 1:
            if 'Adj Close' not in data.columns:
                return pd.DataFrame()
            return data[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
        else:
            if 'Adj Close' not in data.columns:
                return pd.DataFrame()
            return data['Adj Close']
            
    except Exception as e:
        return pd.DataFrame()

def calculate_portfolio_analysis_safe(start_date, end_date):
    """Calculate portfolio analysis with maximum error handling"""
    if not st.session_state.portfolio:
        return None, None, "Portfolio is empty"
    
    try:
        # Validate date range
        if start_date >= end_date:
            return None, None, "Start date must be before end date"
        
        # Check if end date is in future
        if end_date > date.today():
            return None, None, "End date cannot be in the future"
        
        # Prepare data for analysis
        tickers = [stock['ticker'] for stock in st.session_state.portfolio.values()]
        weights = [stock['weight'] for stock in st.session_state.portfolio.values()]
        
        # Convert weights to fractions (0-1)
        total_weight = sum(weights)
        if total_weight <= 0:
            return None, None, "Total portfolio weight must be greater than 0"
        
        weights = [w / total_weight for w in weights]
        
        # Download data with caching
        data = download_portfolio_data_safe(tickers, start_date, end_date)
        
        if data.empty:
            return None, None, "No historical data available for the selected stocks and period"
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        if returns.empty:
            return None, None, "No return data available after processing"
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Validate the returns data
        if len(portfolio_returns) < 5:
            return None, None, "Insufficient data points for analysis (need at least 5 trading days)"
        
        # Ensure datetime index for quantstats
        if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
            portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
        
        return portfolio_returns, data, "Success"
        
    except Exception as e:
        return None, None, f"Error in portfolio analysis: {str(e)}"

def generate_simple_report(portfolio_returns):
    """Generate a simple text report if quantstats fails"""
    try:
        report = []
        report.append("# Portfolio Analysis Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Analysis Period: {len(portfolio_returns)} trading days")
        report.append("")
        report.append("## Key Metrics:")
        report.append(f"- Sharpe Ratio: {qs.stats.sharpe(portfolio_returns):.3f}")
        report.append(f"- Max Drawdown: {qs.stats.max_drawdown(portfolio_returns) * 100:.2f}%")
        report.append(f"- CAGR: {qs.stats.cagr(portfolio_returns) * 100:.2f}%")
        report.append(f"- Volatility: {qs.stats.volatility(portfolio_returns) * 100:.2f}%")
        report.append(f"- Total Return: {qs.stats.comp(portfolio_returns) * 100:.2f}%")
        
        return "\n".join(report), "Success"
    except Exception as e:
        return None, f"Error generating simple report: {str(e)}"

# --- Main Application ---
def main():
    # --- Sidebar: Stock Search Engine & Portfolio Management ---
    with st.sidebar:
        st.header("🔍 NSE Stock Search Engine")
        st.markdown("---")
        
        # Load stock list
        stocks_df = load_stock_list()
        
        if stocks_df.empty:
            st.error("❌ Cannot load stock database. Please check your internet connection.")
            return
        
        # Search Section
        query = st.text_input(
            "Search any NSE stock:",
            placeholder="Type symbol or company name...",
            key="search_input"
        )
        
        if query and len(query) >= 2:
            results = search_stocks(query, stocks_df)
            
            if not results.empty:
                st.success(f"Found {len(results)} stocks")
                
                for _, row in results.iterrows():
                    symbol = row["Symbol"]
                    company = row["Company"]
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{company}**")
                        st.write(f"`{symbol}`")
                    
                    with col2:
                        price, _, _, _ = get_live_price_safe(symbol)
                        if price:
                            st.write(f"₹{price:.1f}")
                        else:
                            st.write("—")
                    
                    with col3:
                        if symbol not in st.session_state.portfolio:
                            if st.button("➕", key=f"add_{symbol}"):
                                if add_to_portfolio(symbol, 0, company):
                                    st.rerun()
                        else:
                            st.button("✅", key=f"added_{symbol}", disabled=True)
                    
                    st.markdown("---")
            else:
                st.warning("No stocks found. Try different keywords.")
        
        # Portfolio Section
        st.header("📊 Your Portfolio")
        st.markdown("---")
        
        if st.session_state.portfolio:
            total_weight = sum(stock['weight'] for stock in st.session_state.portfolio.values())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Stocks", len(st.session_state.portfolio))
            with col2:
                st.metric("Total Weight", f"{total_weight:.1f}%")
            
            for symbol, stock_data in st.session_state.portfolio.items():
                st.write(f"**{symbol}** - {stock_data['company']}")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_weight = st.slider(
                        f"Weight %",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(stock_data['weight']),
                        key=f"weight_{symbol}",
                        step=1.0
                    )
                    if new_weight != stock_data['weight']:
                        update_portfolio_weight(symbol, new_weight)
                
                with col2:
                    if st.button("🗑️", key=f"remove_{symbol}"):
                        remove_from_portfolio(symbol)
                        st.rerun()
                
                st.markdown("---")
            
            # Weight management
            if abs(total_weight - 100.0) > 0.1:
                st.warning(f"⚠️ Weights sum to {total_weight:.1f}% (should be 100%)")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Normalize", use_container_width=True):
                        if normalize_portfolio_weights():
                            st.rerun()
                
                with col2:
                    if st.button("🎯 Equal Weights", use_container_width=True):
                        equal_weight = 100.0 / len(st.session_state.portfolio)
                        for symbol in st.session_state.portfolio:
                            st.session_state.portfolio[symbol]['weight'] = equal_weight
                        st.session_state.analysis_generated = False
                        st.rerun()
            
            if st.button("🗑️ Clear Portfolio", type="secondary", use_container_width=True):
                st.session_state.portfolio = {}
                st.session_state.analysis_generated = False
                st.rerun()
        
        else:
            st.info("👆 Search for stocks above and add them to your portfolio.")
            st.markdown("💡 **Quick start:** Try searching 'RELIANCE', 'TCS', or 'HDFCBANK'")
    
    # --- Main Area: Portfolio Analytics ---
    st.title("📊 NSE Portfolio Analytics Dashboard")
    st.markdown("Build your portfolio using the search engine in the sidebar →")
    
    # Date Range Selection
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        start_date = st.date_input("Start Date", value=date(2023, 1, 1), key="start_date")
    
    with col2:
        end_date = st.date_input("End Date", value=date.today(), key="end_date")
    
    # Reset analysis when dates change
    if 'last_start_date' not in st.session_state:
        st.session_state.last_start_date = start_date
    if 'last_end_date' not in st.session_state:
        st.session_state.last_end_date = end_date
    
    if (st.session_state.last_start_date != start_date or 
        st.session_state.last_end_date != end_date):
        st.session_state.analysis_generated = False
        st.session_state.last_start_date = start_date
        st.session_state.last_end_date = end_date
    
    with col3:
        st.write("")
        st.write("")
        if st.button("🚀 Generate Portfolio Analysis", 
                   type="primary", 
                   use_container_width=True,
                   disabled=len(st.session_state.portfolio) == 0):
            if not st.session_state.portfolio:
                st.error("Please add stocks to your portfolio first!")
            else:
                total_weight = sum(stock['weight'] for stock in st.session_state.portfolio.values())
                if abs(total_weight - 100.0) > 0.1:
                    st.error(f"❌ Portfolio weights must sum to 100% (current: {total_weight:.1f}%)")
                elif start_date >= end_date:
                    st.error("❌ Start date must be before end date")
                elif end_date > date.today():
                    st.error("❌ End date cannot be in the future")
                else:
                    st.session_state.analysis_generated = True
                    st.rerun()
    
    # Portfolio Analysis Results
    if st.session_state.analysis_generated and st.session_state.portfolio:
        with st.spinner("🔄 Analyzing portfolio performance..."):
            portfolio_returns, stock_data, analysis_msg = calculate_portfolio_analysis_safe(start_date, end_date)
            
            if portfolio_returns is not None:
                st.success("✅ Portfolio Analysis Complete!")
                
                # Performance Metrics
                st.subheader("📊 Portfolio Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                try:
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
                except Exception as e:
                    st.error(f"Error calculating some metrics: {str(e)}")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Portfolio Allocation")
                    allocation_data = {
                        'Symbol': list(st.session_state.portfolio.keys()),
                        'Weight': [stock['weight'] for stock in st.session_state.portfolio.values()]
                    }
                    
                    fig_pie = px.pie(
                        allocation_data,
                        values='Weight',
                        names='Symbol',
                        hole=0.3,
                        title="Portfolio Weight Distribution (%)"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    st.subheader("Cumulative Returns")
                    try:
                        cumulative_returns = (1 + portfolio_returns).cumprod()
                        fig_line = px.line(
                            x=cumulative_returns.index,
                            y=cumulative_returns.values,
                            title="Portfolio Growth (₹1 Investment)",
                            labels={'x': 'Date', 'y': 'Value (₹)'}
                        )
                        fig_line.update_layout(showlegend=False)
                        st.plotly_chart(fig_line, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating chart: {str(e)}")
                
                # Download Report
                st.subheader("📥 Download Report")
                try:
                    simple_report, report_msg = generate_simple_report(portfolio_returns)
                    if simple_report:
                        st.download_button(
                            label="Download Portfolio Report (TXT)",
                            data=simple_report,
                            file_name="portfolio_report.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
            
            else:
                st.error(f"❌ Analysis failed: {analysis_msg}")
                st.info("💡 Try adjusting your portfolio or date range and try again.")
    
    elif not st.session_state.portfolio:
        st.info("👈 Start by searching and adding stocks to your portfolio in the sidebar.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Data from NSE India & Yahoo Finance • Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

# Run the application
if __name__ == "__main__":
    main()

