# Semantic-Search

This project implements a semantic search engine for electronic devices using models from the sentence-transformers library from Huggingface for embeddings and Qdrant as the vector database. 

## System Requirements

- Python 3.10
- Docker
- Internet connection for downloading dependencies and models.

## Setup Instructions

### Step 1: Clone the Repository

Clone this repository to your local machine using Git:

```bash
git https://github.com/ahemyu/Semantic-Search.git
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

