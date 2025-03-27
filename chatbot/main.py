#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_backend_server():
    """Run the Flask backend server."""
    try:
        from backend.app import app
        
        port = int(os.environ.get("PORT", 5000))
        debug = os.environ.get("ENVIRONMENT") == "development"
        
        logger.info(f"Starting backend server on port {port}, debug={debug}")
        app.run(host="0.0.0.0", port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Error starting backend server: {str(e)}")
        sys.exit(1)

def run_frontend_app():
    """Run the Streamlit frontend application."""
    try:
        frontend_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "frontend",
            "frontend_streamlit.py"
        )
        
        logger.info(f"Starting Streamlit frontend from {frontend_path}")
        
        # Run streamlit as a subprocess
        subprocess.run(["streamlit", "run", frontend_path], check=True)
        
    except Exception as e:
        logger.error(f"Error starting frontend application: {str(e)}")
        sys.exit(1)

def index_context():
    """Index the context data."""
    try:
        from backend.rag_module import load_and_index_sample_context
        
        logger.info("Indexing context data...")
        load_and_index_sample_context()
        logger.info("Context indexing completed!")
        
    except Exception as e:
        logger.error(f"Error indexing context data: {str(e)}")
        sys.exit(1)

def main():
    """Main entry point for the chatbot application."""
    parser = argparse.ArgumentParser(description="Canada Innovation Strategy Chatbot")
    
    parser.add_argument(
        "--mode",
        choices=["backend", "frontend", "index", "all"],
        default="all",
        help="Mode to run the application in"
    )
    
    args = parser.parse_args()
    
    if args.mode == "backend":
        run_backend_server()
    elif args.mode == "frontend":
        run_frontend_app()
    elif args.mode == "index":
        index_context()
    elif args.mode == "all":
        # First, index the context data
        index_context()
        
        # Then start the frontend app, which will use the backend directly
        run_frontend_app()
    else:
        logger.error(f"Invalid mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main() 