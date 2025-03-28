import os
import sys

# Fix for SQLite version issues (must come before any chromadb import)
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Add the repository root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main Streamlit app
from chatbot.frontend.frontend_streamlit import main

if __name__ == "__main__":
    main() 