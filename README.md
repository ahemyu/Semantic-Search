# Semantic-Search

This project implements a semantic search engine for electronic devices using models from the sentence-transformers library from Huggingface for embeddings and Qdrant as the vector database. 
You can either use the semantic search directly to see results in a table format, or toggle the AI Assistant mode to have an LLM provide conversational responses to your queries (internally also calling semantic search for enhanced context).

## Table of Contents
- [System Requirements](#system-requirements)
- [File Descriptions](#file-descriptions)
  - [data/](#data)
  - [src/](#src)
  - [root](#root)
- [Setup Instructions](#setup-instructions)
  - [Step 1: Clone the Repository](#step-1-clone-the-repository)
  - [Step 2: Install Python Dependencies](#step-2-install-python-dependencies)
  - [Step 3: Start Qdrant Vector Database](#step-3-start-qdrant-vector-database)
  - [Step 4: Choose a model (optional)](#step-4-choose-a-model-optional)
  - [Step 5: Configure LLM (optional)](#step-5-configure-llm-optional)
  - [Step 6: Launch the application](#step-6-launch-the-application)

## System Requirements

- Python 3.10
- Docker
- Internet connection for downloading dependencies and models
- Google Gemini API key (optional, for AI Assistant functionality)

## File Descriptions  
### data/  

The data/ directory serves as the repository for all datasets and processed data files used by the Semantic Search application. 

### src/  
- **data_preprocessing.py**:  

    This script is responsible for loading and preprocessing the data necessary for the semantic search.  

    It imports data from the CSV file in the data directory and  cleans up the text fields by performing tasks like:

    - Lowercasing and stripping whitespaces.  
    - Removing HTML tags and special characters.  
    - Eliminating stopwords that do not contribute to semantic meaning.  
    - Lemmatizing the words to reduce them to their base or dictionary form.  

    The script focuses on columns with descriptive text, combining them into a single column to prepare for embedding. Once processed, the data is saved back to a new CSV file, cleaned_electronic_devices.csv, in the data directory

- **embedding.py** :  

    This script is responsible for generating and managing embeddings used in the semantic search.  
    It utilizes the SentenceTransformer library to load a pre-trained model and create embeddings from the preprocessed text. The functions within the script perform the following operations:

    - load_model: Loads a pre-defined model from the Sentence Transformers library based on the specified model name.  
    - generate_embeddings: Reads in the cleaned_electronic_devices.csv, reads the combined text columns, and uses the loaded model to generate embeddings.  
    - run_embedding: This function orchestrates the creation of a new collection in the Qdrant vector database, uploads the generated embeddings, and associates them with their respective payloads, which include product names, categories, and technical attributes.

- **search.py**:   

    This script handles the semantic search functionality of the project, utilizing both Sentence Transformers for generating query vectors and the Qdrant client for querying the vector database.
  
    - SemanticSearch Class: This class initializes with a model for embeddings, Qdrant client details, and collection name.
    - generate_query_vector: Generates a vector from the input query text using the loaded model, which is then used to search the Qdrant database.
    - search: This function takes a query text and an optional limit to define how many results to return. It performs the actual search in the Qdrant database, retrieving the most semantically similar entries based on the cosine similarity of the embeddings.
    - format_results: Processes the raw search results from Qdrant to format them in a more user-friendly manner, presenting essential details like product category, name, and specifications.  

- **llm_search.py**:

    This script implements the AI Assistant functionality by integrating a Large Language Model with the semantic search capabilities. It utilizes Google's Gemini model through LangChain to provide conversational responses based on search results.

    - LLMSearchAgent Class: This class combines the semantic search functionality with LLM capabilities, initializing with a SemanticSearch instance and Google Gemini API credentials.
    - _semantic_search_tool: Creates a tool function that the LLM agent can call to perform semantic searches, formatting the results for the language model to process and respond to.
    - _create_agent: Sets up the LangChain agent with the semantic search tool, defining the system prompt that instructs the LLM to act as a helpful shopping assistant for electronic products.
    - search_and_respond: Processes user queries through the LLM agent, which intelligently decides when to use the semantic search tool and provides natural language responses with product recommendations and alternatives.

- **main.py**:  

    This script serves as the main entry point for the Semantic Search application, tying together data preprocessing, embedding generation, search functionalities, and the AI Assistant into a comprehensive UI.   
    - load_config: Loads the configuration settings from a JSON file, with overrides from environment variables where applicable. This function sets up essential parameters such as the model name and vector dimensions.
    - setup_environment: Handles the preprocessing of the dataset, generates embeddings with the chosen model, initializes the semantic search system with the specified Qdrant vector database settings, and optionally sets up the LLM agent if API credentials are available.
    - semantic_search_only: Provides direct semantic search functionality that returns results in a DataFrame format for tabular display.
    - llm_search: Handles LLM-enhanced search queries, providing conversational responses through the AI Assistant.
    - Gradio Interface: The user interface is built using Gradio Blocks, featuring a toggle between semantic search mode and AI Assistant mode. Users can switch between viewing raw search results in a table or receiving conversational responses from the LLM, with the interface dynamically updating based on the selected mode.

### root  
- **config.json**:  

    Stores settings that dictate how the application processes and queries data.  
    The parameters included in this configuration file are:

    - data_file_path: Specifies the path to the dataset used for generating embeddings. This ensures that the application knows where to fetch the raw data from within the project structure.
    - model_name: Defines the model used by Sentence Transformers to generate embeddings. This setting can be dynamically overridden by an environment variable.
    - vector_dimension: Indicates the dimensionality of the embeddings generated by the specified model.
    - collection_name: Names the collection within the Qdrant vector database where the embeddings and their associated metadata are stored.
    - qdrant_host and qdrant_port: These settings specify the network address and port for accessing the Qdrant vector database, allowing the application to connect and perform operations on the database.
    - search_limit: Sets the default number of results to return for each search query, which can be adjusted to display more or fewer results as needed.

- **.env** (optional):

    Environment file for storing sensitive configuration like API keys. When using the AI Assistant functionality, create this file in the root directory and add your Google Gemini API key:
    ```
    GEMINI_API_KEY=your_api_key_here
    ```

- **requirements.txt**:  

    The requirements.txt file is used for managing the Python package dependencies required by the Semantic Search project.  
    It lists all the libraries and their specific versions to ensure that the environment is consistent and the application is compatible with python 3.10 as specified.   

## Setup Instructions

### Step 1: Clone the Repository

Clone this repository to your local machine using Git:

```bash
git clone https://github.com/ahemyu/Semantic-Search.git
cd Semantic-Search
```
### Step 2: Install Python Dependencies

Please make sure that you are using python 3.10 as some of the packages won't work with higher/lower versions. 
Also I recommend using a virtual environment like so:

On Linux/Mac: 
```bash
python3 -m venv venv
source venv/bin/activate
```
On Windows: 
```bash
python3 -m venv venv
.\venv\Scripts\activate
```
Then install the required dependencies like so: 

```bash
pip install -r requirements.txt
```

### Step 3: Start Qdrant Vector Database

**For Linux/macOS**

Pull the latest Qdrant Docker image and start the Qdrant server:

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```
**For Windows**

Pull the latest Qdrant Docker image and start the Qdrant server. Ensure Docker Desktop is running and the directory is shared:

- Ensure Docker Desktop is running: Start Docker Desktop from the Start menu or Windows search bar.

- Share the directory:
    - Open Docker Desktop.
    - Go to Settings -> Resources -> File Sharing.
    - Add your project directory (the directory where the project is cloned) and apply the changes.

Run the Docker command in Command Prompt:
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 -v %cd%/qdrant_storage:/qdrant/storage qdrant/qdrant
```
Or in PowerShell:
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant
```
After that, the database will be accessible via a Web UI at: http://localhost:6333/dashboard

### Step 4: Choose a model (optional) 
You can select the model to use by setting the `MODEL_NAME` environment variable. Based on your selection, the application will automatically configure the appropriate vector dimension.  
These are the models you can choose from: 

- all-MiniLM-L12-v2  (384 dimensional embeddings)
- all-MiniLM-L6-v2  (384 dimensional embeddings)
- msmarco-distilbert-base-v3  (768 dimensional embeddings)
- nli-mpnet-base-v2  (768 dimensional embeddings)
- multi-qa-mpnet-base-cos-v1 (768 dimensional embeddings)

On Linux/Mac:
```bash
export MODEL_NAME='model_name'
```
On Windows (Command Prompt):
```bash
set MODEL_NAME=model_name
```
Replace 'model_name' with one of the model names from the list above. 
If you don't set an environment variable yourself the model **all-MiniLM-L6-v2** will be used by default.

**Note**: 
The best performing model according to research and my own testing is **multi-qa-mpnet-base-cos-v1**

### Step 5: Configure LLM (optional)

To enable the AI Assistant functionality, you need to obtain a Google Gemini API key and configure it in your environment.

**Getting a Gemini API Key:**
1. Visit the [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the generated API key

**Setting up the API Key:**

Create a `.env` file in the root directory of the project and add your API key:
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

Alternatively, you can set it as an environment variable:

On Linux/Mac:
```bash
export GEMINI_API_KEY='your_api_key_here'
```
On Windows (Command Prompt):
```bash
set GEMINI_API_KEY=your_api_key_here
```

**Note**: 
If no API key is provided, the application will still function but only in semantic search mode. The AI Assistant toggle will be disabled in the interface.

### Step 6: Launch the application
Go into the src directory and run the main.py like so (might take a while until it runs):  

```bash
cd src
python main.py
```

Now at http://127.0.0.1:7860/ there will be an interface where you can:
- Enter search queries for electronic products
- Toggle between semantic search mode (tabular results) and AI Assistant mode (conversational responses)
