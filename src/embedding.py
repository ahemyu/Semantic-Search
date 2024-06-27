from sentence_transformers import SentenceTransformer
import pandas as pd
from qdrant_client import QdrantClient, models
from typing import List

def load_model(model_name: str) -> SentenceTransformer:
    """Load and return the sentence transformer model based on the model name."""

    return SentenceTransformer(model_name)

def generate_embeddings(clean_file_path: str, model: SentenceTransformer) -> pd.DataFrame:
    """Generate embeddings of the combined text columns"""
    
    df = pd.read_csv(clean_file_path, delimiter=';')
    df['embeddings'] = df['combined'].apply(lambda x: model.encode(x, convert_to_tensor=True).tolist())
    
    return df

def run_embedding(file_path: str, model_name: str, vector_dim: int, collection_name: str, db_url: str) -> None:
    """Create collection in Db and upload the embeddings along with their payloads"""
    
    model = load_model(model_name)
    df = generate_embeddings(file_path, model)
    client = QdrantClient(url=db_url)
    vector_config = models.VectorParams(size=vector_dim, distance=models.Distance.COSINE)
    
    try:
        if not client.collection_exists(collection_name=collection_name):
            client.create_collection(collection_name=collection_name, vectors_config=vector_config)
    except Exception as e:
        print(f"Some Error occurred while creating collection: {e}")
    
    for index, row in df.iterrows():
        point = models.PointStruct(
            id=index,
            vector=row['embeddings'],
            payload={'Product Name': row['productName'], 'Product Category': row['Product_Category'], 'Specifications': row['Technical_Attributes']}
        )
        client.upsert(collection_name=collection_name, points=[point])
