from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM
import pandas as pd

# Create sample DataFrame
sales_by_country = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "sales": [5000, 3200, 2900, 4100, 2300, 2100, 2500, 2600, 4500, 7000]
})

# Initialize the Ollama LLM
ollama_llm = LocalLLM(api_base="http://localhost:11434/v1", model="qwen2.5-coder:32b")

# Create SmartDataframe with Ollama
smart_df = SmartDataframe(sales_by_country, config={"llm": ollama_llm})

# Test the setup with a simple query
if __name__ == "__main__":
    response = smart_df.chat("Which are the top 3 countries by sales?")
    print("Response:", response)
