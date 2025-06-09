import os
import json
import logging
from typing import Dict, Any, List
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

def format_history_for_prompt(history: List[Dict[str, str]], max_turns: int = 3) -> str:
    """Formats the last few turns of chat history for the prompt."""
    formatted_history = []
    # Get the last N * 2 messages (N turns)
    start_index = max(0, len(history) - max_turns * 2) 
    for message in history[start_index:]:
        role = "User" if message["role"] == "user" else "Assistant"
        formatted_history.append(f"{role}: {message['content']}")
    return "\n".join(formatted_history)

def construct_prompt(user_query: str, context_docs: list, history: List[Dict[str, str]] = None) -> str:
    """
    Construct a prompt for the Gemini model with user query, retrieved context, and chat history.
    
    Args:
        user_query: The user's question
        context_docs: The retrieved context documents
        history: List of previous chat messages (optional)
        
    Returns:
        A formatted prompt string
    """
    persona = (
        "You are a helpful AI assistant knowledgeable about Canada's Economy including its innovation strategy and policies, based on analysis from the project. You provide concise and accurate information based on the provided context and conversation history." # Slightly updated persona
    )
    
    context_text = "\n\n".join([doc["content"] for doc in context_docs])
    
    # Format history if available
    formatted_history = ""
    if history:
        # Include the last, say, 3 turns (3 user + 3 assistant messages)
        formatted_history = format_history_for_prompt(history, max_turns=3) 
        
    # Construct the full prompt, including history
    prompt = f"""{persona}

PREVIOUS CONVERSATION:
{formatted_history if formatted_history else "No previous conversation history available."}

CONTEXT INFORMATION RETRIEVED FOR THE CURRENT QUERY:
{context_text}

CURRENT USER QUERY: {user_query}

INSTRUCTIONS:
1. Answer the CURRENT USER QUERY based on the PREVIOUS CONVERSATION and the CONTEXT INFORMATION provided above. Prioritize the CONTEXT INFORMATION if there's a conflict, but use the PREVIOUS CONVERSATION to understand pronoun references (like 'it', 'they', 'each') or follow-up questions.
2. If the context and history don't contain information to answer the query, politely state that the information isn't available in the project findings.
3. Be concise and accurate when asked very specific, direct questions.
4. If the user asks for details or the question implies elaboration, be expansive and provide as much relevant information as possible while remaining accurate, drawing from both context and conversation history.
5. Do not mention that you are using 'context', 'history', or that your knowledge comes from specific documents. Just provide the answer naturally.

YOUR RESPONSE:"""
    
    return prompt

def handle_chat_query(user_query: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        logger.info(f"Processing chat query: {user_query}")
        if history:
            logger.info(f"Received history with {len(history)} messages.")

        context_docs = rag_manager.retrieve_relevant_context(user_query, top_k=5)
        
        if not context_docs:
            logger.warning("No relevant context found for query")
            return {
                "answer": "I don't have specific information about that in my knowledge base. Could you try asking about Canada's innovation strategy, R&D factors, or policy recommendations?",
                "sources": []
            }
        
        prompt = construct_prompt(user_query, context_docs, history)
        response = get_gemini_response(prompt)
        
        # Updated sources preparation
        sources = []
        for doc in context_docs:
            valid_topics = []
            if "topics" in doc["metadata"] and isinstance(doc["metadata"]["topics"], list):
                valid_topics = [
                    topic for topic in doc["metadata"]["topics"]
                    if isinstance(topic, dict) and "confidence" in topic and "name" in topic
                ]
            
            topic_name = "general"
            if valid_topics:
                try:
                    # Find the topic with the highest confidence
                    best_topic = max(valid_topics, key=lambda x: x["confidence"])
                    topic_name = best_topic["name"]
                except (ValueError, TypeError): 
                    # Handle potential errors if confidence is not comparable or list is empty after filtering
                    logger.warning(f"Could not determine best topic for doc id {doc.get('id', 'N/A')}, defaulting to 'general'. Topics: {doc['metadata'].get('topics', 'N/A')}")
                    topic_name = "general"

            sources.append({
                "id": doc.get("id", "N/A"), # Use .get for safety
                "topic": topic_name
            })

        logger.info("Successfully generated response for chat query")
        return {
            "answer": response,
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Error handling chat query: {str(e)}", exc_info=True)
        return {
            "answer": "I'm sorry, I encountered an error while processing your request.",
            "sources": []
        }

# Modify API endpoint to accept history
@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """API endpoint for chat functionality."""
    try:
        data = request.json
        user_query = data.get('query', '')
        history = data.get('history', []) # Get history from payload, default to empty list
        
        if not user_query:
            return jsonify({"error": "No query provided"}), 400
        
        # Pass both query and history to the handler
        response = handle_chat_query(user_query, history) 
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True) # Added exc_info
        return jsonify({"error": "Internal server error"}), 500 # Avoid leaking error details

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

# Run the Flask app if executed directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("ENVIRONMENT") == "development") 