from sentence_transformers import SentenceTransformer
import pandas as pd
from qdrant_client import QdrantClient, models

def load_model(model_name):
    """Load and return the sentence transformer model based on the model name."""
    return SentenceTransformer(model_name)

def generate_embeddings(df, model):
    """Generate embeddings by combining the text columns and encoding them using the model."""
    combined_text = df['Description'] + " " + df['Marketing_Text'] + " " + df['Typical_Use_Cases'] + " " + df['Technical_Attributes']
    return combined_text.apply(lambda x: model.encode(x, convert_to_tensor=True).tolist())

## config
file_path = '../data/cleaned_electronic_devices.csv'
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
vector_dimension = 384
collection_name = 'electronic_products'
db_url = "http://localhost:6333"


## main 
try:
    df = pd.read_csv(file_path, delimiter=';')
    model = load_model(model_name)
    df['embeddings'] = generate_embeddings(df, model)
    client = QdrantClient(url=db_url)

    vector_config = models.VectorParams(size=vector_dimension, distance=models.Distance.COSINE)
    if collection_name not in client.get_collections().collections:
        client.create_collection(collection_name=collection_name, vectors_config=vector_config)

    for index, row in df.iterrows():
        point = models.PointStruct(
            id=index,
            vector=row['embeddings'],
            payload={'Product Name': row['productName'], 'Product Category': row['Product_Category']}
        )
        client.upsert(collection_name=collection_name, points=[point])
except Exception as e:
    print(f"An error occurred: {e}")
