import os
import gradio as gr
import pandas as pd
import json
from typing import Dict, Any
from data_preprocessing import preprocess_data
from embedding import run_embedding
from search import SemanticSearch
from llm_search import LLMSearchAgent
from dotenv import load_dotenv

load_dotenv()

# choosable models for the embeddings
model_dimensions = {
    'all-MiniLM-L12-v2': 384,
    'all-MiniLM-L6-v2': 384,
    'msmarco-distilbert-base-v3': 768,
    'nli-mpnet-base-v2': 768,
    "multi-qa-mpnet-base-cos-v1": 768,
    "all-mpnet-base-v2": 768,
}


def load_config() -> Dict[str, Any]:
    """Load configuration settings from environment variables or fall back to config.json."""
    
    with open('../config.json', 'r') as config_file:
        config = json.load(config_file)
    
    # Model configuration
    model_name = os.getenv('MODEL_NAME', config['model_name'])
    config['model_name'] = model_name
    config['vector_dimension'] = model_dimensions.get(
        model_name, config['vector_dimension']
    )
        
    return config


config = load_config()


def setup_environment() -> tuple[SemanticSearch, LLMSearchAgent | None]:
    """Setup the environment by preprocessing data, generating embeddings, and initializing the search engine"""
    
    cleaned_data_path = preprocess_data(config["data_file_path"])
    run_embedding(
        cleaned_data_path,
        config["model_name"],
        config["vector_dimension"],
        config["collection_name"],
        f"http://{config['qdrant_host']}:{config['qdrant_port']}"
    )
    
    searcher = SemanticSearch(
        config["model_name"],
        config["qdrant_host"],
        config["qdrant_port"],
        config["collection_name"]
    )
    
    # Initialize LLM agent if API key is available
    llm_agent = LLMSearchAgent(searcher, os.getenv('GEMINI_API_KEY')) if os.getenv('GEMINI_API_KEY') else None
    
    return searcher, llm_agent


searcher, llm_agent = setup_environment()


def search_products(query: str, use_llm: bool = True) -> str | pd.DataFrame:
    """Search for products using either direct search or LLM-enhanced search"""
    
    if use_llm and llm_agent:
        # Use LLM for natural language response
        return llm_agent.search_and_respond(query)
    else:
        # Fall back to direct search
        results = searcher.search(query, limit=config.get('search_limit', 5))
        return pd.DataFrame(results)


# Create Gradio interface
if llm_agent:
    # Semantic Search with LLM
    iface = gr.Interface(
        fn=lambda query: search_products(query, use_llm=True),
        inputs=gr.Textbox(
            label="What are you looking for?",
            lines=2
        ),
        outputs=gr.Textbox(label="Assistant Response", lines=10),
        title=f"AI Shopping Assistant",
        description="Ask me about electronic products and I'll help you find what you need!",
        examples=[
            ["I am looking for a gaming laptop"],
            ["I need a laptop to do light work with."],
            ["cheap smartphone"],
            ["High performance desktop pc"],
        ],
        allow_flagging='never'
    )
else:
    # Just semantic search
    iface = gr.Interface(
        fn=lambda query, limit: search_products(query, use_llm=False),
        inputs=[
            gr.Textbox(label="Search Query", placeholder="Enter your query"),
            gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of Results")
        ],
        outputs="dataframe",
        title=f"Semantic Search (Using Model: {config['model_name']})",
        description="Enter your query to search for products",
        allow_flagging='never'
    )

iface.launch()