import os
import json
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pathlib import Path

from .gemini_interface import get_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive environment)
    BASE_DIR = Path('.').resolve()

# ChromaDB settings
DEFAULT_CHROMA_PATH = BASE_DIR / "chatbot" / "data" / "chroma_db"
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", str(DEFAULT_CHROMA_PATH))
logger.info(f"Using ChromaDB directory: {CHROMA_DB_DIR}")

DEFAULT_COLLECTION_NAME = "canada_innovation_strategy"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", DEFAULT_COLLECTION_NAME)
logger.info(f"Using ChromaDB collection name: {COLLECTION_NAME}")

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

        logger.info(f"Initializing RAGManager with DB: '{self.db_directory}', Collection: '{self.collection_name}'")

        
        # Ensure DB directory exists
        try:
            Path(self.db_directory).mkdir(parents=True, exist_ok=True) # Use pathlib
        except Exception as e:
            logger.error(f"Failed to create ChromaDB directory '{self.db_directory}': {e}", exc_info=True)
            raise
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=self.db_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"ChromaDB client initialized for path: '{self.db_directory}'")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client for path '{self.db_directory}': {e}", exc_info=True)
            raise
        
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
            logger.info(f"Attempting to get collection: '{self.collection_name}'")
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.ef # Pass embedding function here for consistency
            )
            logger.info(f"Retrieved existing collection: '{self.collection_name}'")
        except Exception as e_get:
            logger.warning(f"Collection '{self.collection_name}' not found ({e_get}), attempting to create.")
            try:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.ef # Pass embedding function here
                )
                logger.info(f"Successfully created new collection: '{self.collection_name}'")
            except Exception as e_create:
                logger.error(f"Failed to create collection '{self.collection_name}': {e_create}", exc_info=True)
                raise
    
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
            try:
                embedding = get_embeddings(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error getting embedding for text snippet: {e}", exc_info=True)
                # Decide handling: skip, use zero vector, or raise? Raising for now.
                raise
        return embeddings
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of document dictionaries with id, content, and metadata
        """
        if not documents:
            logger.warning("add_documents called with empty document list.")
            return
        try:
            ids = [doc["id"] for doc in documents]
            contents = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            logger.info(f"Attempting to add {len(documents)} documents to collection '{self.collection_name}'...")

            
            # Handle embeddings differently based on embedding function choice
            if self.use_gemini_embeddings:
                logger.info("Generating embeddings via Gemini...")

                # Generate embeddings using Gemini
                embeddings = self._embed_with_gemini(contents)
                logger.info("Embeddings generated. Adding documents with embeddings...")
                
                # Add documents with pre-computed embeddings
                self.collection.add(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:
                # Add documents, let ChromaDB handle embeddings
                logger.info("Adding documents (embeddings handled by Chroma)...")

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
        logger.info(f"Attempting to retrieve context for query in collection '{self.collection_name}'")
        try:
            results = None # Initialize results

            if self.use_gemini_embeddings:
                logger.info("Generating query embedding via Gemini...")

                # Get query embedding using Gemini
                query_embedding = get_embeddings(query)
                logger.info("Query embedding generated. Querying collection...")

                # Query with pre-computed embedding
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=filter_metadata
                )
            else:
                # Let ChromaDB handle the embedding
                logger.info("Querying collection with ChromaDB...")

                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where=filter_metadata
                )
            
            # Format the results
            if not results or not results.get("documents") or not isinstance(results["documents"], list) or not results["documents"]:
                 logger.warning(f"Received empty or invalid results structure from ChromaDB query for collection '{self.collection_name}'.")
                 return []
            if len(results["documents"]) == 0 or len(results["documents"][0]) == 0:
                logger.warning(f"ChromaDB query returned no documents for collection '{self.collection_name}'.")
                return []


            # Format the results (with safety checks)
            documents = []
            num_results = len(results["documents"][0])
            ids = results["ids"][0] if results.get("ids") and results["ids"] else [None] * num_results
            metadatas = results["metadatas"][0] if results.get("metadatas") and results["metadatas"] else [{}] * num_results
            distances = results["distances"][0] if results.get("distances") and results["distances"] else [None] * num_results
            contents = results["documents"][0] # Already checked this exists

            for i in range(num_results):
                 documents.append({
                    "id": ids[i],
                    "content": contents[i],
                    "metadata": metadatas[i] if metadatas[i] else {}, # Ensure metadata is a dict
                    "distance": distances[i]
                 })


            logger.info(f"Retrieved {len(documents)} relevant documents from collection '{self.collection_name}'")
            return documents

        except Exception as e:
            # Log the specific error during retrieval
            logger.error(f"Error retrieving context from collection '{self.collection_name}': {e}", exc_info=True)
            # Check for specific ChromaDB errors if possible, e.g., collection not found
            # Re-raising for now, could return [] depending on desired behavior
            # raise # Re-raising might hide the warning in app.py, returning empty list might be better
            return [] # Return empty list on error to allow app.py warning
        

def _load_context_file(file_path: Path) -> List[Dict[str, Any]]: # Use Path
    """
    Load a context file and return its contents. (Added logging)

    Args:
        file_path: Path object for the context file

    Returns:
        List of context documents
    """
    logger.info(f"Attempting to load context file: '{file_path}'")
    try:
        with file_path.open('r', encoding='utf-8') as f:
            documents = json.load(f)
        logger.info(f"Successfully loaded {len(documents)} documents from '{file_path}'")
        return documents
    except FileNotFoundError:
        logger.error(f"Context file not found: '{file_path}'")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from context file '{file_path}': {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading context file '{file_path}': {e}", exc_info=True)
        return []

def load_and_index_sample_context() -> None:
    """
    Utility function to load context from JSON and index it. (Updated pathing, config, logging)
    """
    logger.info("--- Starting Context Indexing Process ---")
    try:
        # Use configured paths and names
        data_dir = BASE_DIR / "chatbot" / "data" # Use pathlib path
        generated_context_path = data_dir / "generated_context.json"
        sample_context_path = data_dir / "sample_context.json"

        documents = []
        if generated_context_path.exists():
            logger.info(f"Found generated context file: '{generated_context_path}'")
            documents = _load_context_file(generated_context_path)
        elif sample_context_path.exists():
            logger.warning(f"Generated context not found, attempting to load sample context from '{sample_context_path}'")
            documents = _load_context_file(sample_context_path)
        else:
            logger.error(f"Neither generated nor sample context file found in '{data_dir}'. Cannot index.")
            return

        if not documents:
            logger.error("No context documents could be loaded. Aborting indexing.")
            return

        # Initialize the RAG manager (uses configured path/name from env vars/defaults)
        logger.info("Initializing RAGManager for indexing...")
        rag_manager = RAGManager()

        # Delete existing collection if it exists (with logging)
        try:
            logger.warning(f"Attempting to delete existing collection: '{rag_manager.collection_name}' before re-indexing.")
            rag_manager.client.delete_collection(rag_manager.collection_name)
            logger.info(f"Successfully deleted existing collection: '{rag_manager.collection_name}'")
            # Re-create the collection explicitly after deletion
            logger.info(f"Re-creating collection: '{rag_manager.collection_name}'")
            rag_manager.collection = rag_manager.client.create_collection(
                name=rag_manager.collection_name,
                embedding_function=rag_manager.ef
            )
            logger.info(f"Successfully re-created collection: '{rag_manager.collection_name}'")
        except Exception as e:
            logger.warning(f"Could not delete collection '{rag_manager.collection_name}' (may not exist yet): {e}")
            # Ensure collection exists if deletion failed
            if not rag_manager.collection:
                 logger.info(f"Attempting to get or create collection '{rag_manager.collection_name}' after failed deletion attempt.")
                 rag_manager.collection = rag_manager.client.get_or_create_collection(
                     name=rag_manager.collection_name,
                     embedding_function=rag_manager.ef
                 )
                 logger.info(f"Ensured collection '{rag_manager.collection_name}' exists.")


        # Add the documents to the vector database
        rag_manager.add_documents(documents)

        logger.info(f"--- Context Indexing Process Completed Successfully ({len(documents)} documents indexed) ---")

    except Exception as e:
        logger.error(f"--- Error during Context Indexing Process ---: {e}", exc_info=True)
        # Decide if we should raise the exception or just log it
        # raise # Optional: re-raise if indexing failure should stop the app

logger.debug(f"BASE_DIR calculated as: {BASE_DIR}")
