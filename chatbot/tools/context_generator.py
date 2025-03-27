#!/usr/bin/env python3
import os
import re
import json
import argparse
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
import nltk
from nltk.tokenize import sent_tokenize
import PyPDF2
from docx import Document
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from a Word document.
    
    Args:
        file_path: Path to the Word document
        
    Returns:
        Extracted text as a string
    """
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from Word document {file_path}: {str(e)}")
        return ""

def extract_text_from_jupyter(file_path: str) -> str:
    """
    Extract text from a Jupyter notebook, focusing on markdown cells.
    
    Args:
        file_path: Path to the Jupyter notebook
        
    Returns:
        Extracted text as a string
    """
    text = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            notebook = json.load(file)
        
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                markdown_text = ''.join(cell['source'])
                text += markdown_text + "\n\n"
            elif cell['cell_type'] == 'code':
                # You can optionally include code comments or code output
                if 'outputs' in cell and len(cell['outputs']) > 0:
                    for output in cell['outputs']:
                        if 'text' in output:
                            text += ''.join(output['text']) + "\n\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from Jupyter notebook {file_path}: {str(e)}")
        return ""

def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text from a plain text file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Extracted text as a string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except UnicodeDecodeError:
        # Try with a different encoding if utf-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from text file {file_path}: {str(e)}")
            return ""
    except Exception as e:
        logger.error(f"Error extracting text from text file {file_path}: {str(e)}")
        return ""

def extract_text_from_file(file_path: str) -> Optional[Tuple[str, str]]:
    """
    Extract text from a file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (extracted text, source name) or None if unsupported file type
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    source_name = os.path.splitext(file_name)[0]
    
    if file_ext == '.pdf':
        return extract_text_from_pdf(file_path), source_name
    elif file_ext == '.docx':
        return extract_text_from_docx(file_path), source_name
    elif file_ext == '.ipynb':
        return extract_text_from_jupyter(file_path), source_name
    elif file_ext == '.txt':
        return extract_text_from_txt(file_path), source_name
    else:
        logger.warning(f"Unsupported file type: {file_ext}")
        return None

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks of approximately chunk_size characters,
    trying to break at paragraph or sentence boundaries.
    
    Args:
        text: The text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # First try to split by paragraphs (double newlines)
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            # If paragraph is too big for one chunk, split by sentences
            if not current_chunk:  # Handle case where paragraph is bigger than chunk_size
                sentences = sent_tokenize(para)
                temp_chunk = ""
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) <= chunk_size:
                        temp_chunk += sentence + " "
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence + " "
                
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Add overlap if needed
    if overlap > 0 and len(chunks) > 1:
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]
            
            # Try to include some text from previous chunk if possible
            overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
            result.append(overlap_text + curr_chunk)
        
        return result
    
    return chunks

def extract_topics(chunk: str) -> List[str]:
    """
    Extract potential topics from a text chunk using simple keyword matching.
    
    Args:
        chunk: The text chunk
        
    Returns:
        List of identified topics
    """
    topics = []
    
    # Define keyword mappings for topics
    topic_keywords = {
        "innovation_metrics": ["innovation index", "ranking", "performance", "metrics", "indicators"],
        "r&d_intensity": ["r&d", "research and development", "spending", "investment", "gdp"],
        "policy_analysis": ["policy", "plan", "strategy", "government", "federal", "program"],
        "regional_policies": ["provincial", "region", "quebec", "ontario", "british columbia", "alberta"],
        "talent_factors": ["talent", "skills", "education", "workforce", "recruitment", "retention"],
        "funding_factors": ["funding", "venture capital", "investment", "financing", "capital"],
        "sentiment_results": ["sentiment", "perception", "opinion", "coverage", "media"],
        "recommendations": ["recommend", "suggestion", "improve", "enhance", "future", "strategy"]
    }
    
    # Check for each topic's keywords in the chunk
    chunk_lower = chunk.lower()
    for topic, keywords in topic_keywords.items():
        if any(keyword in chunk_lower for keyword in keywords):
            topics.append(topic)
    
    # If no topics found, use "general"
    if not topics:
        topics.append("general")
    
    return topics

def create_context_data(extracted_texts: List[Tuple[str, str]], 
                       output_path: str,
                       chunk_size: int = 1000,
                       overlap: int = 200) -> None:
    """
    Create context data for the chatbot from extracted texts.
    
    Args:
        extracted_texts: List of tuples containing (text, source_name)
        output_path: Path to save the generated context data
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
    """
    context_data = []
    
    for text, source_name in extracted_texts:
        # Chunk the text
        chunks = chunk_text(text, chunk_size, overlap)
        
        for i, chunk in enumerate(chunks):
            # Skip very short chunks
            if len(chunk) < 100:
                continue
                
            # Extract topics
            topics = extract_topics(chunk)
            primary_topic = topics[0] if topics else "general"
            
            # Create unique ID
            chunk_id = f"{source_name.lower().replace(' ', '_')}_{primary_topic}_{i+1}"
            
            # Add to context data
            context_data.append({
                "id": chunk_id,
                "content": chunk,
                "metadata": {
                    "source": source_name,
                    "topic": primary_topic
                }
            })
    
    # Write to JSON file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(context_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully created context data with {len(context_data)} chunks at {output_path}")
    except Exception as e:
        logger.error(f"Error writing context data to {output_path}: {str(e)}")

def main():
    """Main function to run the context generator."""
    parser = argparse.ArgumentParser(description="Generate context data for the Canada Innovation Strategy Chatbot")
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../input_documents",
        help="Directory containing input documents (PDF, DOCX, IPYNB, TXT)"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="../data/generated_context.json",
        help="Output JSON file path for the generated context data"
    )
    
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Target size of each text chunk in characters"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Number of characters to overlap between chunks"
    )
    
    args = parser.parse_args()
    
    # Create input directory if it doesn't exist
    os.makedirs(args.input_dir, exist_ok=True)
    
    # Get all files in input directory
    files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) 
             if os.path.isfile(os.path.join(args.input_dir, f))]
    
    logger.info(f"Found {len(files)} files in {args.input_dir}: {files}")
    
    if not files:
        logger.warning(f"No files found in {args.input_dir}")
        return
    
    # Extract text from all files
    extracted_texts = []
    for file_path in files:
        logger.info(f"Processing file: {file_path}")
        result = extract_text_from_file(file_path)
        if result:
            text, source_name = result
            logger.info(f"Successfully extracted text from {file_path}, source: {source_name}")
            extracted_texts.append((text, source_name))
        else:
            logger.warning(f"Failed to extract text from {file_path}")
    
    logger.info(f"Extracted text from {len(extracted_texts)} files")
    
    if not extracted_texts:
        logger.warning("No text could be extracted from the provided files")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create context data
    create_context_data(
        extracted_texts=extracted_texts,
        output_path=args.output_file,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )

if __name__ == "__main__":
    main() 