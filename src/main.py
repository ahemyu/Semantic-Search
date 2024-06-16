import gradio as gr
import pandas as pd
import json
from data_preprocessing import preprocess_data
from embedding import run_embedding
from search import SemanticSearch

def load_config():
    with open('../config.json', 'r') as config_file:
        return json.load(config_file)

config = load_config()

def setup_environment():
    cleaned_data_path = preprocess_data(config["data_file_path"])
    run_embedding(cleaned_data_path, config["model_name"], config["vector_dimension"], config["collection_name"], f"http://{config['qdrant_host']}:{config['qdrant_port']}")
    searcher = SemanticSearch(config["model_name"], config["qdrant_host"], config["qdrant_port"], config["collection_name"])
    return searcher

searcher = setup_environment()

def search_products(query):
    results = searcher.search(query, limit=config["search_limit"])
    data = [{
        'Name': res['productName'],
        'Category': res['productCategory'],
        'Specifications': res['specifications']
    } for res in results]
    
    return pd.DataFrame(data)

iface = gr.Interface(
    fn=search_products,
    inputs="text",
    outputs="dataframe",
    title="Product Search",
    description="Enter your query"
)
iface.launch()
