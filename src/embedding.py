from sentence_transformers import SentenceTransformer
import pandas as pd
from qdrant_client import QdrantClient, models

def load_model(model_name):
    """Load and return the sentence transformer model based on the model name."""
    
    return SentenceTransformer(model_name)

def generate_embeddings(file_path, model):
    """Generate embeddings by combining text columns and encoding ."""
    df = pd.read_csv(file_path, delimiter=';')
    combined_text = df['productName'] + " "+ df['Product_Category'] + " " + df['Description'] + " " + df['Marketing_Text'] + " " + df['Typical_Use_Cases'] + " " + df['Technical_Attributes'] # maybe add name and catgeory as well? 
    df['embeddings'] = combined_text.apply(lambda x: model.encode(x, convert_to_tensor=True).tolist())
    return df

def run_embedding(file_path, model_name, vector_dim, collection_name, db_url):
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

if __name__ == "__main__":
    
    run_embedding('../data/cleaned_electronic_devices.csv', 'sentence-transformers/all-MiniLM-L6-v2', 384, 'electronic_products', "http://localhost:6333")