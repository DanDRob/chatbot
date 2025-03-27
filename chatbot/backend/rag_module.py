import os
import json
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from .gemini_interface import get_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ChromaDB settings
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
COLLECTION_NAME = "canada_innovation_strategy"

class RAGManager:
    def __init__(
        self, 
        collection_name: str = COLLECTION_NAME,
        db_directory: str = CHROMA_DB_DIR,
        use_gemini_embeddings: bool = True
    ):
        """
        Initialize the RAG Manager to handle vector database operations.
        
        Args:
            collection_name: Name of the vector DB collection
            db_directory: Directory to store the ChromaDB data
            use_gemini_embeddings: Whether to use Gemini or open-source embeddings
        """
        self.collection_name = collection_name
        self.db_directory = db_directory
        self.use_gemini_embeddings = use_gemini_embeddings
        
        # Ensure DB directory exists
        os.makedirs(self.db_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.db_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Set up embedding function
        if not self.use_gemini_embeddings:
            # Use sentence-transformers for embeddings
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        else:
            # Use Gemini embeddings (custom function)
            self.ef = None  # We'll use the get_embeddings function directly
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.ef
            )
            logger.info(f"Retrieved existing collection: {self.collection_name}")
        except Exception:
            logger.info(f"Creating new collection: {self.collection_name}")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.ef
            )
    
    def _embed_with_gemini(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings using Gemini API for multiple texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = get_embeddings(text)
            embeddings.append(embedding)
        return embeddings
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of document dictionaries with id, content, and metadata
        """
        try:
            ids = [doc["id"] for doc in documents]
            contents = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            # Handle embeddings differently based on embedding function choice
            if self.use_gemini_embeddings:
                # Generate embeddings using Gemini
                embeddings = self._embed_with_gemini(contents)
                
                # Add documents with pre-computed embeddings
                self.collection.add(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:
                # Add documents, let ChromaDB handle embeddings
                self.collection.add(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas
                )
                
            logger.info(f"Successfully added {len(documents)} documents to collection")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector database: {str(e)}")
            raise
    
    def retrieve_relevant_context(
        self, 
        query: str, 
        top_k: int = 3, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant context for a query.
        
        Args:
            query: The user query
            top_k: Number of documents to retrieve
            filter_metadata: Optional filter to apply based on metadata
            
        Returns:
            List of relevant documents
        """
        try:
            if self.use_gemini_embeddings:
                # Get query embedding using Gemini
                query_embedding = get_embeddings(query)
                
                # Query with pre-computed embedding
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=filter_metadata
                )
            else:
                # Let ChromaDB handle the embedding
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where=filter_metadata
                )
            
            # Format the results
            documents = []
            for i in range(len(results["documents"][0])):
                documents.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
            
            logger.info(f"Retrieved {len(documents)} relevant documents for query")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving context from vector database: {str(e)}")
            return []

def _load_context_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a context file and return its contents.
    
    Args:
        file_path: Path to the context file
        
    Returns:
        List of context documents
    """
    try:
        with open(file_path, 'r') as f:
            documents = json.load(f)
        logger.info(f"Successfully loaded {len(documents)} documents from {file_path}")
        return documents
    except Exception as e:
        logger.error(f"Error loading context file {file_path}: {str(e)}")
        return []

def load_and_index_sample_context() -> None:
    """
    Utility function to load context from JSON files and index it to the vector database.
    First tries to load from generated_context.json, then falls back to sample_context.json.
    """
    try:
        # Get the paths to the context files
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        generated_context_path = os.path.join(data_dir, "generated_context.json")
        sample_context_path = os.path.join(data_dir, "sample_context.json")
        
        # Try to load generated context first, fall back to sample context
        if os.path.exists(generated_context_path):
            logger.info(f"Loading generated context from {generated_context_path}")
            documents = _load_context_file(generated_context_path)
        else:
            logger.info(f"Generated context not found, loading sample context from {sample_context_path}")
            documents = _load_context_file(sample_context_path)
        
        if not documents:
            logger.error("No context documents could be loaded")
            return
        
        # Initialize the RAG manager
        rag_manager = RAGManager()
        
        # Delete existing collection if it exists
        try:
            rag_manager.client.delete_collection(rag_manager.collection_name)
            logger.info(f"Deleted existing collection: {rag_manager.collection_name}")
            # Recreate the collection
            rag_manager.collection = rag_manager.client.create_collection(
                name=rag_manager.collection_name,
                embedding_function=rag_manager.ef
            )
        except Exception as e:
            logger.info(f"No existing collection to delete: {str(e)}")
        
        # Add the documents to the vector database
        rag_manager.add_documents(documents)
        
        logger.info(f"Successfully indexed {len(documents)} documents to vector database")
        
    except Exception as e:
        logger.error(f"Error loading and indexing context: {str(e)}")
        raise 