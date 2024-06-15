# Semantic-Search

This project implements a semantic search engine for electronic devices using the sentence transformer **all-MiniLM-L6-v2** from Huggingface for embeddings and Qdrant as the vector database. 

## System Requirements

- Python 3.10
- Docker
- Internet connection for downloading dependencies and models.

## Setup Instructions

### Step 1: Clone the Repository

Clone this repository to your local machine using Git:

```bash
git clone git@github.com:ahemyu/Semantic-Search.git
cd Semantic-Search
```
### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Start Qdrant Vector Database

Pull the latest Qdrant docker image and start the Qdrant server:
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```
After that the Database will be accessible via a Web UI at:  http://localhost:6333/dashboard


### Step 4: Launch the application
Go into the src directory and run the main.py like so:  

```bash
cd src
python3 main.py
```

Now at http://127.0.0.1:7860/ there will be a simple Interface where a search query can be put in 
