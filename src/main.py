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


def semantic_search_only(query: str, limit: int = 5) -> pd.DataFrame:
    """Perform semantic search and return results as DataFrame"""
    results = searcher.search(query, limit=limit)
    return pd.DataFrame(results)


def llm_search(query: str) -> str:
    """Perform LLM-enhanced search"""
    if not llm_agent:
        return "LLM functionality is not available. Please provide a Gemini API key."
    return llm_agent.search_and_respond(query)


# Create Gradio interface with Blocks for more control
with gr.Blocks(title="Electronic Products Search") as iface:
    gr.Markdown(f"# Electronic Products Search\n### Embedding Model: {config['model_name']}")
    if llm_agent:
        gr.Markdown("### LLM: gemini-2.5-flash-preview-05-20")
    
    # Add toggle only if LLM is available
    if llm_agent:
        use_llm = gr.Checkbox(
            label="Use AI Assistant", 
            value=True,
            info="Toggle between semantic search results and AI-powered responses"
        )
    else:
        use_llm = gr.Checkbox(
            label="Use AI Assistant (Unavailable - No API Key)", 
            value=False,
            interactive=False,
            info="Gemini API key required for AI Assistant"
        )
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="Enter what you're looking for...",
                lines=2
            )
            
            # Slider for result limit (only shown for semantic search)
            limit_slider = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Number of Results",
                visible=not (llm_agent and use_llm.value)
            )
            
            search_btn = gr.Button("Search", variant="primary")
            
    with gr.Row():
        # Output components
        semantic_output = gr.DataFrame(
            label="Search Results",
            visible=not (llm_agent and use_llm.value)
        )
        llm_output = gr.Textbox(
            label="AI Assistant Response",
            lines=15,
            visible=(llm_agent and use_llm.value)
        )
    
    def update_interface(use_llm_value):
        """Update interface based on toggle state"""
        if use_llm_value and llm_agent:
            return {
                limit_slider: gr.update(visible=False),
                semantic_output: gr.update(visible=False),
                llm_output: gr.update(visible=True)
            }
        else:
            return {
                limit_slider: gr.update(visible=True),
                semantic_output: gr.update(visible=True),
                llm_output: gr.update(visible=False)
            }
    
    def perform_search(query, use_llm_value, limit):
        """Perform search based on selected mode"""
        if use_llm_value and llm_agent:
            result = llm_search(query)
            return {
                semantic_output: gr.update(value=None),
                llm_output: gr.update(value=result)
            }
        else:
            result = semantic_search_only(query, limit)
            return {
                semantic_output: gr.update(value=result),
                llm_output: gr.update(value=None)
            }
    
    # Event handlers
    use_llm.change(
        update_interface,
        inputs=[use_llm],
        outputs=[limit_slider, semantic_output, llm_output]
    )
    
    search_btn.click(
        perform_search,
        inputs=[query_input, use_llm, limit_slider],
        outputs=[semantic_output, llm_output]
    )
    
    query_input.submit(
        perform_search,
        inputs=[query_input, use_llm, limit_slider],
        outputs=[semantic_output, llm_output]
    )

iface.launch()