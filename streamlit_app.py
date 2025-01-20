import streamlit as st
from pandasai import SmartDataframe
from pandasai.callbacks import BaseCallback
from pandasai.llm import LangchainLLM
from pandasai.responses.response_parser import ResponseParser
from langchain_community.llms import Ollama

from data import load_data


class StreamlitCallback(BaseCallback):
    def __init__(self, container) -> None:
        """Initialize callback handler."""
        self.container = container

    def on_code(self, response: str):
        self.container.code(response)


class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        try:
            st.dataframe(result["value"])
        except Exception as e:
            st.error(f"Error displaying dataframe: {str(e)}")
        return

    def format_plot(self, result):
        try:
            if isinstance(result["value"], str):
                st.image(result["value"])
            else:
                st.pyplot(result["value"])
        except Exception as e:
            st.error(f"Error displaying plot: {str(e)}")
        return

    def format_other(self, result):
        try:
            if isinstance(result, dict) and "value" in result:
                st.write(result["value"])
            else:
                st.write(result)
        except Exception as e:
            st.error(f"Error displaying result: {str(e)}")
        return

    def parse(self, response):
        try:
            if isinstance(response, (str, int, float, bool)):
                return self.format_other({"value": response})
            elif isinstance(response, dict):
                if "type" not in response:
                    return self.format_other(response)
                if response["type"] == "plot":
                    return self.format_plot(response)
                elif response["type"] == "dataframe":
                    return self.format_dataframe(response)
                else:
                    return self.format_other(response)
            else:
                return self.format_other({"value": str(response)})
        except Exception as e:
            st.error(f"Error parsing response: {str(e)}")
            return None


st.write("# Chat with Dataset ")

df = load_data("./data")

with st.expander(" Dataframe Preview"):
    st.write(df.tail(3))

query = st.text_area(" Chat with Dataframe")
response_container = st.container()
plot_container = st.container()

if query:
    ollama = Ollama(base_url="http://localhost:11434", model="qwen2.5-coder:32b")
    llm = LangchainLLM(llm=ollama)
    query_engine = SmartDataframe(
        df,
        config={
            "llm": llm,
            "response_parser": StreamlitResponse,
            "callback": StreamlitCallback(response_container),
            "save_charts": True,
            "save_charts_path": "./cache/charts",
        },
    )

    with response_container:
        st.write("")
    answer = query_engine.chat(query)
    with response_container:
        st.empty()
        st.write("")
import os

import streamlit as st
from pandasai import SmartDataframe
from pandasai.callbacks import BaseCallback
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser

from data import load_data


class StreamlitCallback(BaseCallback):
    def __init__(self, container) -> None:
        """Initialize callback handler."""
        self.container = container

    def on_code(self, response: str):
        self.container.code(response)


class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return


st.write("# Chat with Dataset 🦙")

df = load_data("./data")

with st.expander("🔎 Dataframe Preview"):
    st.write(df.tail(3))

query = st.text_area("🗣️ Chat with Dataframe")
container = st.container()

if query:
    llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"])
    query_engine = SmartDataframe(
        df,
        config={
            "llm": llm,
            "response_parser": StreamlitResponse,
            "callback": StreamlitCallback(container),
        },
    )

    answer = query_engine.chat(query)