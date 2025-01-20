from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM
import pandas as pd

# Load the financial data
df = pd.read_csv('data/myport2.csv', parse_dates=['Date'], date_format='%Y%m%d')
df.set_index('Date', inplace=True)

# Initialize the Ollama LLM
ollama_llm = LocalLLM(api_base="http://localhost:11434/v1", model="qwen2.5-coder:32b")

# Create SmartDataframe with Ollama
smart_df = SmartDataframe(df, config={"llm": ollama_llm})

# Test the setup with a simple query
if __name__ == "__main__":
    try:
        response = smart_df.chat("Show the latest values of DHC, FMC, and REE along with their dates.")
        print("Response:", response)
    except Exception as e:
        print("An error occurred during execution:", e)
