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

# --- Model and Config Definitions ---

model_dimensions = {
    "all-MiniLM-L12-v2": 384,
    "all-MiniLM-L6-v2": 384,
    "msmarco-distilbert-base-v3": 768,
    "nli-mpnet-base-v2": 768,
    "multi-qa-mpnet-base-cos-v1": 768,
    "all-mpnet-base-v2": 768,
}

AVAILABLE_LLMS = {
    "Gemini 2.5 Flash (Google)": {
        "provider": "google",
        "model": "gemini-2.5-flash-preview-05-20",
        "api_key": os.getenv("GEMINI_API_KEY"),
    },
    "GPT-4o (OpenAI)": {
        "provider": "openai",
        "model": "gpt-4o-2024-08-06",
        "api_key": os.getenv("OPENAI_API_KEY"), 
    },
    "Claude 4.0 Sonnet (Anthropic)": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
    },
}


def load_config() -> Dict[str, Any]:
    """Load configuration settings."""
    with open("../config.json", "r") as config_file:
        config = json.load(config_file)
    model_name = os.getenv("MODEL_NAME", config["model_name"])
    config["model_name"] = model_name
    config["vector_dimension"] = model_dimensions.get(
        model_name, config["vector_dimension"]
    )
    return config


# --- Application Setup ---

config = load_config()


def setup_environment() -> SemanticSearch:
    """
    Preprocess data and generate embeddings if the model has changed,
    then initialize the search engine.
    """
    last_used_model = config.get("last_used_model", "")
    current_model = config["model_name"]

    if last_used_model != current_model:
        print(
            f"Embedding model changed from '{last_used_model}' to '{current_model}'."
        )
        print("Regenerating embeddings and creating new collection...")

        cleaned_data_path = preprocess_data(config["data_file_path"])
        run_embedding(
            cleaned_data_path,
            current_model,
            config["vector_dimension"],
            config["collection_name"],
            f"http://{config['qdrant_host']}:{config['qdrant_port']}",
        )

        # Update the config file with the new model name
        config_path = "../config.json"
        try:
            with open(config_path, "r+") as f:
                config_data = json.load(f)
                config_data["last_used_model"] = current_model
                f.seek(0)
                json.dump(config_data, f, indent=4)
                f.truncate()
            print(
                f"Configuration updated. Current embedding model is '{current_model}'."
            )
        except Exception as e:
            print(f"Error updating config file '{config_path}': {e}")

    else:
        print(f"Using existing collection with model '{current_model}'.")

    return SemanticSearch(
        config["model_name"],
        config["qdrant_host"],
        config["qdrant_port"],
        config["collection_name"],
    )


def initialize_llm_agents(
    searcher: SemanticSearch,
) -> Dict[str, LLMSearchAgent]:
    """Pre-initializes all LLM agents for which an API key is available."""
    agents = {}
    for name, details in AVAILABLE_LLMS.items():
        if details["api_key"]:
            print(f"Initializing agent: {name}")
            agents[name] = LLMSearchAgent(
                searcher=searcher,
                llm_provider=details["provider"],
                model_name=details["model"],
                api_key=details["api_key"],
            )
    return agents


# --- Initialize Core Components ---

searcher = setup_environment()
llm_agents = initialize_llm_agents(searcher)
enabled_llms = list(llm_agents.keys())
is_llm_available = bool(enabled_llms)


# --- Search Functions ---


def semantic_search_only(query: str, limit: int = 5) -> pd.DataFrame:
    """Perform semantic search and return results as DataFrame."""
    results = searcher.search(query, limit=limit)
    return pd.DataFrame(results)


def llm_search(
    query: str, selected_llm: str, history: list
) -> tuple[str, list]:
    """Perform LLM-enhanced search using a pre-initialized agent."""
    if not selected_llm or selected_llm not in llm_agents:
        error_msg = "Error: Please select a valid and initialized LLM."
        return error_msg, history

    agent = llm_agents[selected_llm]
    response = agent.search_and_respond(query, history)

    # Update history for the next turn
    history.append([query, response])
    return response, history


# --- Gradio UI ---

with gr.Blocks(title="Electronic Products Search") as iface:
    gr.Markdown(
        f"# Electronic Products Search\n### Embedding Model: {config['model_name']}"
    )

    # State to hold conversation history for the selected LLM
    chat_history_state = gr.State([])

    if is_llm_available:
        with gr.Row():
            use_llm = gr.Checkbox(
                label="Use AI Assistant", value=True, scale=1
            )
            llm_selector = gr.Dropdown(
                label="Select AI Model",
                choices=enabled_llms,
                value=enabled_llms[0],
                scale=2,
            )
    else:
        use_llm = gr.Checkbox(
            label="Use AI Assistant (Unavailable)",
            value=False,
            interactive=False,
        )
        llm_selector = gr.Dropdown(visible=False)

    # Outputs
    semantic_output = gr.DataFrame(label="Search Results", visible=False)
    llm_output = gr.Textbox(
        label="AI Assistant Response", lines=15, visible=is_llm_available
    )

    # Inputs
    with gr.Row():
        query_input = gr.Textbox(
            label="Search Query",
            placeholder="Enter what you're looking for...",
            lines=2,
            scale=4,
        )
        search_btn = gr.Button("Search", variant="primary", scale=1)

    limit_slider = gr.Slider(
        minimum=1,
        maximum=20,
        value=5,
        step=1,
        label="Number of Results",
        visible=False,
    )

    # --- UI Logic ---

    def update_ui_mode(use_llm_value):
        """Toggle visibility of UI components based on search mode."""
        show_llm = use_llm_value and is_llm_available
        return {
            llm_selector: gr.update(visible=show_llm),
            llm_output: gr.update(visible=show_llm),
            semantic_output: gr.update(visible=not show_llm),
            limit_slider: gr.update(visible=not show_llm),
        }

    def handle_search(query, use_llm_value, limit, selected_llm, history):
        """Main function to route search requests."""
        if use_llm_value and is_llm_available:
            response, updated_history = llm_search(query, selected_llm, history)
            return response, pd.DataFrame(), updated_history
        else:
            results_df = semantic_search_only(query, limit)
            return "", results_df, history  # History remains unchanged

    def clear_chat():
        """Clears the chat history and output when switching models."""
        return [], ""

    # --- Event Handlers ---

    if is_llm_available:
        use_llm.change(
            fn=update_ui_mode,
            inputs=use_llm,
            outputs=[
                llm_selector,
                llm_output,
                semantic_output,
                limit_slider,
            ],
        )
        llm_selector.change(
            fn=clear_chat, inputs=None, outputs=[chat_history_state, llm_output]
        )

    search_btn.click(
        fn=handle_search,
        inputs=[
            query_input,
            use_llm,
            limit_slider,
            llm_selector,
            chat_history_state,
        ],
        outputs=[llm_output, semantic_output, chat_history_state],
    )

iface.launch()