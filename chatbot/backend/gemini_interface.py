import os
import logging
import numpy as np
from typing import Optional, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("GOOGLE_API_KEY")
USE_MOCK = api_key is None or api_key == "your_google_api_key_here"
if USE_MOCK:
    logger.warning("No valid API key found. Using mock responses for development purposes.")
else:
    # Configure Gemini API with real key
    genai.configure(api_key=api_key)

def get_gemini_response(
    prompt: str, 
    model_name: str = "gemini-1.5-flash-latest",
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
    safety_settings: Optional[Dict[str, Any]] = None
) -> str:
    """
    Get a response from the Gemini API.
    
    Args:
        prompt: The prompt to send to the model
        model_name: The name of the Gemini model to use
        temperature: Controls randomness (0.0-1.0)
        max_output_tokens: Maximum output length
        safety_settings: Custom safety settings if needed
        
    Returns:
        The generated text response
    """
    try:
        logger.info(f"Sending prompt to Gemini API using model: {model_name}")
        
        # Configure the generation model
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        
        # Get the model
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Log success and return response text
        logger.info("Successfully received response from Gemini API")
        return response.text
        
    except Exception as e:
        # Log error and return error message
        error_msg = f"Error getting response from Gemini API: {str(e)}"
        logger.error(error_msg)
        return f"I'm sorry, I encountered an error: {str(e)}"

def get_embeddings(text: str, model_name: str = "models/embedding-001") -> list:
    """
    Get embeddings for a text using the Gemini Embedding API.
    
    Args:
        text: The text to embed
        model_name: The name of the embedding model to use
        
    Returns:
        The embedding vector
    """
    try:
        logger.info(f"Getting embeddings using model: {model_name}")
        
        # Use the updated embeddings API (embedding-001 is the model name)
        embedding = genai.embed_content(
            model=model_name,
            content=text,
            task_type="RETRIEVAL_QUERY"
        )
        
        # Log success and return embedding values
        logger.info("Successfully received embeddings from Gemini API")
        return embedding["embedding"]
        
    except Exception as e:
        # Log error and raise exception
        error_msg = f"Error getting embeddings from Gemini API: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(f"Failed to generate embeddings: {str(e)}") 