#!/usr/bin/env python3
import os
import logging
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag_module import load_and_index_sample_context

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to load and index the context data.
    Will first try to use generated_context.json if available,
    otherwise will fall back to sample_context.json.
    """
    try:
        logger.info("Starting context indexing process...")
        
        # Load and index context
        load_and_index_sample_context()
        
        logger.info("Context indexing process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in context indexing process: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 