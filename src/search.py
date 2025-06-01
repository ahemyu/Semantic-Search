from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from typing import List, Dict, Any

class SemanticSearch:
    def __init__(self, model_name: str, qdrant_host: str, qdrant_port: int, collection_name: str):
        """Initialize the search system with the model and Qdrant client."""

        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name

    def generate_query_vector(self, query_text: str) -> List[float]:
        """Generate a vector from the query text """

        return self.model.encode(query_text, convert_to_tensor=True).tolist()

    def search(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search and return the 'limit' most similar items."""
        
        query_vector = self.generate_query_vector(query_text)
        search_params = models.SearchParams(hnsw_ef=128, exact=False)
        search_results: List[models.ScoredPoint] = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            search_params=search_params,
            limit=limit,
            with_payload=True,
        )

        # Filter out the 'combined' field from the payload
        return [{k: v for k, v in scored_point.payload.items() if k != "combined"} for scored_point in search_results]