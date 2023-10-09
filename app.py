import toml
import streamlit as st
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index import VectorStoreIndex

import os
from pypdf import PdfReader


os.environ["OPENAI_API_KEY"] = st.secrets["secrets"]["OPENAI_API_KEY"]

# @st.cache_resource(experimental_allow_widgets=True)
# def load_data():
#     documents = SimpleDirectoryReader('data').load_data()
#     return GPTVectorStoreIndex.from_documents(documents)

# index = load_data()

# st.title("Rehab Document Query Engine")
# user_input = st.text_input("Ask a question about the AKPMR document:")

# if user_input:
#     result = index.query(user_input)
#     st.write(result)


@st.cache_resource(experimental_allow_widgets=True)
def load_data():
    documents = SimpleDirectoryReader('data').load_data()
    return VectorStoreIndex.from_documents(documents)

index = load_data()
query_engine = index.as_query_engine()  # Create a query engine from the index

st.title("Rehab Document Query Engine")
user_input = st.text_input("Ask a question about the AKPMR document:")

if user_input:
    result = query_engine.query(user_input)  # Use the query method on the query engine object
    st.write(result)




