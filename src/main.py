import gradio as gr
import pandas as pd
from data_preprocessing import preprocess_data
from embedding import run_embedding
from search import SemanticSearch

def setup_environment():
    cleaned_data_path = preprocess_data('../data/Semantic_Search_Test_Set_Electronic_Devices.csv')
    run_embedding(cleaned_data_path, 'sentence-transformers/all-MiniLM-L6-v2', 384, 'electronic_products', "http://localhost:6333")
    searcher = SemanticSearch('sentence-transformers/all-MiniLM-L6-v2', 'localhost', 6333, 'electronic_products')
    return searcher

searcher = setup_environment()

def search_products(query):
    results = searcher.search(query, limit=5)
    data = [{
        'Name': res['productName'],
        'Category': res['productCategory'],
        'Specifications': res['specifications']
    } for res in results]
    # Convert list of dicts to DataFrame
    return pd.DataFrame(data)

iface = gr.Interface(
    fn=search_products,
    inputs="text",
    outputs="dataframe",
    title="Product Search",
    description="Enter your query"
)
iface.launch()