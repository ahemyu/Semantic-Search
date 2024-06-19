# Semantic-Search

This project implements a semantic search engine for electronic devices using models from the sentence-transformers library from Huggingface for embeddings and Qdrant as the vector database. 

## System Requirements

- Python 3.10
- Docker
- Internet connection for downloading dependencies and models.

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

    The script focuses on columns with descriptive text, combining them into a single column to prepare for embedding. Once processed, the data is saved back to a new CSV file, cleaned_electronic_devices.csv, in the data directory, optimized for subsequent embedding and search functionalities.

- **embedding.py** :  
    This script is essential for the generation and management of embeddings used in the semantic search system.  
    It utilizes the SentenceTransformer library to load a pre-trained model and create embeddings from the preprocessed text. The functions within the script perform the following operations:

    - load_model: Loads a pre-defined model from the Sentence Transformers library based on the specified model name.  
    - generate_embeddings: Reads in the cleaned_electronic_devices.csv, reads the combined text columns, and uses the loaded model to generate embeddings.  
    - run_embedding: This main function orchestrates the creation of a new collection in the Qdrant vector database, uploads the generated embeddings, and associates them with their respective payloads, which include product names, categories, and technical attributes. It ensures that if the collection does not exist in the database, it is created with the appropriate vector configuration set for cosine distance measurement.  

- **search.py**:   
    This script handles the semantic search functionality of the project, utilizing both Sentence Transformers for generating query vectors and the Qdrant client for querying the vector database.  
    The key components include:  

    - SemanticSearch Class: This class initializes with a model for embeddings, Qdrant client details, and collection name.
    - generate_query_vector: Generates a vector from the input query text using the loaded model, which is then used to search the Qdrant database.
    - search: This function takes a query text and an optional limit to define how many results to return. It performs the actual search in the Qdrant database, retrieving the most semantically similar entries based on the cosine similarity of the embeddings.
    - format_results: Processes the raw search results from Qdrant to format them in a more user-friendly manner, presenting essential details like product category, name, and specifications.  

- **main.py**:  
    This script serves as the main entry point for the Semantic Search application, tying together data preprocessing, embedding generation, and search functionalities into a simple user interface.   
    The operations are orchestrated through several key functions:  

    - load_config: Loads the configuration settings from a JSON file, with overrides from environment variables where applicable. This function sets up essential parameters such as the model name and vector dimensions.
    - setup_environment: Handles the preprocessing of the dataset, generates embeddings with the chosen model, and initializes the semantic search system with the specified Qdrant vector database settings.
    - search_products: Defines the actual search functionality that is exposed to the user through a Gradio interface. It utilizes the SemanticSearch instance to execute queries and formats the results into a readable format.
    - Gradio Interface: The user interface is set up using Gradio, allowing users to input their search queries and view the results in a tabular format.  
### root directory 
- **config.json**:  
    Stores essential settings that dictate how the application processes and queries data. The parameters included in this configuration file are:

    - data_file_path: Specifies the path to the dataset used for generating embeddings. This ensures that the application knows where to fetch the raw data from within the project structure.
    - model_name: Defines the model used by Sentence Transformers to generate embeddings. This setting can be dynamically overridden by an environment variable if needed.
    - vector_dimension: Indicates the dimensionality of the embeddings generated by the specified model, which is crucial for correctly configuring the vector space in the Qdrant database.
    - collection_name: Names the collection within the Qdrant vector database where the embeddings and their associated metadata are stored.
    - qdrant_host and qdrant_port: These settings specify the network address and port for accessing the Qdrant vector database, allowing the application to connect and perform operations on the database.
    - search_limit: Sets the default number of results to return for each search query, which can be adjusted to display more or fewer results as needed.
- **requirements.txt**:  
    The requirements.txt file is used for managing the Python package dependencies required by the Semantic Search project.  
    It lists all the libraries and their specific versions to ensure that the environment is consistent and the application is compatible with python 3.10 as specified.   
## Setup Instructions

### Step 1: Clone the Repository

Clone this repository to your local machine using Git:

```bash
git https://github.com/ahemyu/Semantic-Search.git
```
cd Semantic-Search
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

### Step 5: Choose a model (optional) 
You can select the model to use by setting the `MODEL_NAME` environment variable. Based on your selection, the application will automatically configure the appropriate vector dimension.
These are the models you can choose from: 

- all-MiniLM-L12-v2  (384 dimensional embeddings)
- all-MiniLM-L6-v2  (384 dimensional embeddings)
- msmarco-distilbert-base-v3  (768 dimensional embeddings)
- nli-mpnet-base-v2  (768 dimensional embeddings)

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
If you want to try different models consider deleting the collection in Qdrant first (very easy through the Web UI), as the dimensions of the embeddings differ per model. 

### Step 5: Launch the application
Go into the src directory and run the main.py like so (might take a while until it runs):  

```bash
cd src
python main.py
```

Now at http://127.0.0.1:7860/ there will be a simple Interface where a search query can be put in 

