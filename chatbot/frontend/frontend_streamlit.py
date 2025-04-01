import os
import sys
import json
import logging
import requests
from typing import Dict, List, Any
import streamlit as st

# Fix for SQLite version issues (must come before chromadb import)
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Add the root directory to the path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import backend modules (for direct integration without API)
from chatbot.backend.rag_module import RAGManager
from chatbot.backend.app import handle_chat_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
API_MODE = os.environ.get("USE_API", "false").lower() == "true"
API_URL = os.environ.get("API_URL", "http://localhost:5000")

# Page configuration
st.set_page_config(
    page_title="Canada Innovation Strategy Chatbot",
    page_icon="🍁",
    layout="centered"
)

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources" not in st.session_state:
        st.session_state.sources = {}
    #if "db_initialized" not in st.session_state:
        # Only initialize the database once on first load
    #    from chatbot.backend.rag_module import load_and_index_sample_context
    #    with st.spinner("Initializing knowledge base (this may take a minute)..."):
    #        load_and_index_sample_context()
    #    st.session_state.db_initialized = True
    #    st.success("Knowledge base initialized successfully!")

def display_chat_header():
    """Display the chat header with title and project information."""
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>Canada Innovation Strategy Chatbot</h1>
            <p>MIE1624 Course Project - Ask me about Canada's innovation landscape and policies!</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("---")

def display_chat_messages():
    """Display the chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🍁"):
            st.markdown(message["content"])
            
            # Display sources if this is an assistant message with sources
            if message["role"] == "assistant" and message.get("message_id") in st.session_state.sources:
                sources = st.session_state.sources[message["message_id"]]
                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.markdown(f"**Topic:** {source['topic']}")

def query_backend_api(user_query: str, history: List[Dict[str, str]]) -> Dict[str, Any]: # Add history parameter
    """
    Query the backend API with a user question and chat history.
    
    Args:
        user_query: The user's question
        history: The list of previous chat messages
        
    Returns:
        Response dictionary with answer and sources
    """
    try:
        payload = {"query": user_query, "history": history} # Include history in payload
        response = requests.post(
            f"{API_URL}/api/chat",
            # json={"query": user_query}, # Old payload
            json=payload,                 # New payload
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error querying backend API: {str(e)}")
        return {
            "answer": f"I'm sorry, I encountered an error communicating with the backend: {str(e)}",
            "sources": []
        }

def main():
    """Main function to run the Streamlit app."""
    initialize_session_state()
    display_chat_header()
    display_chat_messages()
    
    if user_query := st.chat_input("Ask about Canada's innovation strategy..."):
        current_history = st.session_state.messages # Get current history before appending new user message
        
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_query)
        
        with st.chat_message("assistant", avatar="🍁"):
            message_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                if API_MODE:
                    # Pass the history *before* the current user message was added
                    response = query_backend_api(user_query, current_history) 
                else:
                    # Pass the history *before* the current user message was added
                    # NOTE: Need to update handle_chat_query signature in app.py as well
                    response = handle_chat_query(user_query, current_history) # Pass history directly
                
                answer = response.get("answer", "I'm sorry, I couldn't process your request.")
                sources = response.get("sources", [])
            
            message_placeholder.markdown(answer)
        
        message_id = f"msg_{len(st.session_state.messages)}"
        
        # Add assistant message AFTER getting the response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "message_id": message_id
        })
        
        st.session_state.sources[message_id] = sources
        
        # Display sources below the assistant's response if available
        if sources:
            with st.expander("📚 Sources"):
                for source in sources:
                    st.markdown(f"**Topic:** {source['topic']}")

if __name__ == "__main__":
    main() 