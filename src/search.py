from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

class SemanticSearch:
    def __init__(self, model_name, qdrant_host, qdrant_port, collection_name):
        """Initialize the search system with the model and Qdrant client."""
        
        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name

    def generate_query_vector(self, query_text):
        """Generate a vector from the query text """
        
        return self.model.encode(query_text, convert_to_tensor=True).tolist()

    def search(self, query_text, limit=10):
        """Perform semantic search and return the 'limit' most similar items."""
        
        query_vector = self.generate_query_vector(query_text)
        search_params = models.SearchParams(hnsw_ef=128, exact=False)
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            search_params=search_params,
            limit=limit,
            with_payload=True,
        )
        return self.format_results(search_results)

    def format_results(self, search_results):
        """Format search results for frontend display to user"""
        
        formatted_results = []
        for scored_point in search_results:
            formatted_result = {
                'productCategory': scored_point.payload['Product Category'],
                'productName': scored_point.payload['Product Name'],
                'specifications': scored_point.payload.get('Specifications', 'No specifications available')
            }
            formatted_results.append(formatted_result)
            
        return formatted_results