import os
import json
import logging
from typing import Dict, Any
from flask import Flask, request, jsonify

from .gemini_interface import get_gemini_response
from .rag_module import RAGManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize RAG manager - use specific collection_name to match the one from load_and_index_sample_context()
rag_manager = RAGManager(collection_name="canada_innovation_strategy")

def construct_prompt(user_query: str, context_docs: list) -> str:
    """
    Construct a prompt for the Gemini model with user query and retrieved context.
    
    Args:
        user_query: The user's question
        context_docs: The retrieved context documents
        
    Returns:
        A formatted prompt string
    """
    # Define the chatbot persona
    persona = (
        "You are an AI assistant knowledgeable about Canada's innovation strategy, "
        "based on analysis from the MIE1624 project. You provide concise, accurate "
        "information based only on the provided context."
    )
    
    # Format the context information
    context_text = "\n\n".join([doc["content"] for doc in context_docs])
    
    # Construct the full prompt
    prompt = f"""{persona}

CONTEXT INFORMATION:
{context_text}

USER QUERY: {user_query}

INSTRUCTIONS:
1. Answer the user's query based ONLY on the information in the context provided above.
2. If the context doesn't contain information to answer the query, politely state that the information isn't available in the project findings.
3. Be concise and accurate.
4. Do not mention that you are using 'context' or that your knowledge comes from specific documents.

YOUR RESPONSE:"""
    
    return prompt

def handle_chat_query(user_query: str) -> Dict[str, Any]:
    """
    Handle a chat query by retrieving context and generating a response.
    
    Args:
        user_query: The user's question
        
    Returns:
        Response dictionary with answer
    """
    try:
        logger.info(f"Processing chat query: {user_query}")
        
        # Retrieve relevant context from vector database
        context_docs = rag_manager.retrieve_relevant_context(user_query, top_k=3)
        
        if not context_docs:
            logger.warning("No relevant context found for query")
            return {
                "answer": "I don't have specific information about that in my knowledge base. Could you try asking about Canada's innovation strategy, R&D factors, or policy recommendations?",
                "sources": []
            }
        
        # Construct prompt with retrieved context
        prompt = construct_prompt(user_query, context_docs)
        
        # Get response from Gemini
        response = get_gemini_response(prompt)
        
        # Prepare source references
        sources = [{"id": doc["id"], "topic": doc["metadata"]["topic"]} for doc in context_docs]
        
        logger.info("Successfully generated response for chat query")
        return {
            "answer": response,
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Error handling chat query: {str(e)}")
        return {
            "answer": f"I'm sorry, I encountered an error while processing your request: {str(e)}",
            "sources": []
        }

# API Routes
@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """API endpoint for chat functionality."""
    try:
        data = request.json
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({"error": "No query provided"}), 400
        
        response = handle_chat_query(user_query)
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

# Run the Flask app if executed directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("ENVIRONMENT") == "development") 