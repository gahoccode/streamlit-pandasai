import streamlit as st
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv
import pandasai.helpers.cache
import duckdb

# Load environment variables
load_dotenv()

# Create a custom cache class to completely replace the original implementation
class CustomCache:
    def __init__(self):
        self.filepath = ":memory:"
        self.connection = duckdb.connect(self.filepath)
        self.enabled = True
        
        # Create the cache table with the correct schema
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY,
                key TEXT,
                value TEXT,
                prompt TEXT,
                response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def get(self, key):
        result = self.connection.execute("SELECT value FROM cache WHERE key=?", [key]).fetchone()
        if result:
            return result[0]
        return None
    
    def set(self, key, value):
        self.connection.execute(
            "INSERT INTO cache (key, value) VALUES (?, ?)",
            [key, value]
        )
    
    def has(self, key):
        result = self.connection.execute("SELECT 1 FROM cache WHERE key=?", [key]).fetchone()
        return result is not None
    
    def clear(self):
        self.connection.execute("DELETE FROM cache")

# Replace the original Cache class with our custom implementation
original_cache_init = pandasai.helpers.cache.Cache.__init__

def create_cache_instance(*args, **kwargs):
    return CustomCache()

# Monkey patch the Cache class
pandasai.helpers.cache.Cache = CustomCache

# Page config
st.set_page_config(page_title="Portfolio Analysis AI", page_icon="📈", layout="wide")
st.write("# Portfolio Analysis AI 📈")

# Load and prepare data
def load_portfolio_data():
    df = pd.read_csv('data/myport2.csv', parse_dates=['Date'], date_parser=lambda x: pd.to_datetime(x, format='%Y%m%d'))
    df.set_index('Date', inplace=True)
    return df.dropna()

def execute_and_capture_output(code, local_vars):
    # Capture both printed output and returned variables
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    
    result = None
    try:
        exec(code, globals(), local_vars)
        # Look for common visualization variable names
        for var_name in ['fig', 'plt', 'chart', 'plot']:
            if var_name in local_vars:
                result = local_vars[var_name]
                break
    finally:
        sys.stdout = old_stdout
        
    return redirected_output.getvalue(), result

def process_query(query, df):
    if not query:
        return None
    
    # Add visualization request to the query if not already present
    if "visualization" not in query.lower() and "chart" not in query.lower() and "plot" not in query.lower():
        query += " Create a line chart visualization showing this data over time."
    elif "visualization" in query.lower() and "line chart" not in query.lower():
        query = query.replace("visualization", "line chart visualization")
    
    try:
        st.info("Running PandasAI with openai LLM...")
        result = df.chat(query)
        
        # If result is just a string and no visualization, try again with more explicit request
        if isinstance(result, str) and "chart" in query.lower():
            enhanced_query = f"Create a line chart visualization showing {query}. Return the chart object, not just text."
            result = df.chat(enhanced_query)
            
        return result
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Load data
df = load_portfolio_data()

# Sidebar for date range selection
st.sidebar.header("Time Period")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df.index[-365], df.index[-1]),
    min_value=df.index[0],
    max_value=df.index[-1]
)

# Filter data based on date range
mask = (df.index >= pd.Timestamp(date_range[0])) & (df.index <= pd.Timestamp(date_range[1]))
filtered_df = df[mask]

# Display basic portfolio statistics
st.subheader("Portfolio Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Portfolio Size", f"{len(filtered_df.columns)} assets")
with col2:
    st.metric("Time Period", f"{len(filtered_df)} days")
with col3:
    returns = filtered_df.pct_change()
    portfolio_return = returns.mean().mean() * 252  # Annualized return
    st.metric("Annualized Return", f"{portfolio_return:.2%}")

# Initialize PandasAI with OpenAI
def initialize_pandasai(df):
    # Get OpenAI API key from environment variable or Streamlit secrets
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If not found in environment, ask user to input it
    if not api_key:
        api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
        if not api_key:
            st.warning("Please enter your OpenAI API key to use the AI features.")
            return None
    
    # Initialize OpenAI LLM
    llm = OpenAI(api_token=api_key)
    
    # Create SmartDataframe with explicit config for visualizations
    smart_df = SmartDataframe(
        df, 
        config={
            "llm": llm,
            "enable_cache": True,
            "save_charts": True,
            "verbose": True,
            "enforce_privacy": False,
            "save_charts_path": "./",
            "open_charts": False,
            "max_retries": 3,
            "use_error_correction_framework": True
        }
    )
    
    return smart_df

# Initialize PandasAI
smart_df = initialize_pandasai(filtered_df)

if smart_df is None:
    st.stop()

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Price Chart", "Portfolio Analysis", "AI Assistant"])

with tab1:
    st.subheader("Asset Price Evolution")
    fig = px.line(filtered_df, title="Asset Prices Over Time")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Portfolio Statistics")
    returns = filtered_df.pct_change()
    
    # Calculate and display key metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Returns Statistics")
        st.dataframe(returns.describe())
    
    with col2:
        st.write("Correlation Matrix")
        st.dataframe(returns.corr())

with tab3:
    st.subheader("AI Portfolio Assistant")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    query = st.text_input(
        "Ask a question about your portfolio:",
        placeholder="e.g., What's the Sharpe ratio of the portfolio? Show it with visualization",
        key="query_input"
    )
    
    if query:
        with st.spinner('Analyzing...'):
            try:
                response = process_query(query, smart_df)
                
                # Check if response is a string or a visualization
                if isinstance(response, str):
                    st.write("Response:", response)
                else:
                    st.write("Analysis complete! Here's what I found:")
                
                # Display the visualization if available
                if hasattr(response, 'figure') or isinstance(response, go.Figure):
                    st.plotly_chart(response, use_container_width=True)
                elif str(type(response).__module__).startswith('matplotlib'):
                    st.pyplot(response)
                elif isinstance(response, pd.DataFrame):
                    st.dataframe(response)
                
                # Check for saved chart files
                chart_files = [f for f in os.listdir('./') if f.endswith('.png') and f.startswith('temp_chart')]
                for chart_file in chart_files:
                    st.image(chart_file, caption="Generated Visualization")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and PandasAI 🚀")
