from pathlib import Path
import pandas as pd
import streamlit as st

@st.cache_data
def load_data(folder: str = ".") -> pd.DataFrame:
    """
    Load financial CSV files from the specified folder.
    
    Args:
        folder (str): Directory containing CSV files. Defaults to current directory.
    
    Returns:
        pd.DataFrame: Combined dataframe with properly formatted financial data
    """
    # Get all CSV files in the specified folder
    csv_files = list(Path(folder).glob("*.csv"))
    
    if not csv_files:
        st.warning("No CSV files found in the specified directory.")
        return pd.DataFrame()
    
    # Load all CSV files
    all_datasets = []
    for file in csv_files:
        try:
            # Read CSV with proper date parsing
            df = pd.read_csv(file, parse_dates=['Date'], 
                           date_parser=lambda x: pd.to_datetime(x, format='%Y%m%d'))
            
            # Sort by date
            df = df.sort_values('Date')
            
            all_datasets.append(df)
            st.success(f"Successfully loaded {file.name}")
        except Exception as e:
            st.error(f"Error loading {file.name}: {str(e)}")
    
    if not all_datasets:
        st.error("No valid CSV files could be loaded.")
        return pd.DataFrame()
    
    # Combine all datasets and sort by date
    df = pd.concat(all_datasets, ignore_index=True)
    df = df.sort_values('Date')
    
    return df
