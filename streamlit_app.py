import os
import sys

# Add the repository root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main Streamlit app
from chatbot.frontend.frontend_streamlit import main

if __name__ == "__main__":
    main() 