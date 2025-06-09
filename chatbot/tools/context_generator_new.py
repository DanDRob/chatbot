import os
import json
import re
import spacy
from spacy.matcher import PhraseMatcher
import PyPDF2
from nbformat import read as read_notebook
from docx import Document

# Load spaCy model for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Define topics with associated keywords and phrases
topics = {
    "innovation_metrics": [
        "innovation index", "ranking", "performance", "metrics", "indicators",
        "gii", "global innovation index", "benchmark", "rank", "wipo"
        ],
    "r&d_intensity": [
        "r&d", "research and development", "spending", "investment", "gdp",
        "berd", "business expenditure", "research & development"
        ],
    "policy_analysis": [
        "policy", "plan", "strategy", "government", "federal", "program",
        "initiative", "regulation", "deregulation", "legislation", "framework",
        "ita", "innovation tax advantage", "tci", "talent canada initiative",
        "seiia", "strategic energy infrastructure", "sr&ed", "ised", "oitc",
        "isc", "innovative solutions canada", "sbir", "tax credit", "tax break",
        "procurement", "public procurement", "jenkins report", "advisory council"
        ],
    "regional_policies": [
        "provincial", "region", "quebec", "ontario", "british columbia", "alberta",
        "territorial", "regional energy hubs"
        ],
    "talent_factors": [
        "talent", "skills", "education", "workforce", "recruitment", "retention",
        "brain drain", "emigration", "skilled workers", "human capital", "stem",
        "compensation", "wages", "salary", "housing", "global talent stream",
        "express entry", "lmia", "re-entry"
        ],
    "funding_factors": [
        "funding", "venture capital", "investment", "financing", "capital",
        "vc", "seed funding", "series b", "late-stage", "early-stage",
        "capital gains", "subsidy", "grant", "incentive", "matching investment"
        ],
    "recommendations": [
        "recommend", "suggestion", "improve", "enhance", "future", "strategy",
        "proposal", "should", "must", "roadmap", "action", "path"
        ],
    "commercialization": [
        "commercialization", "market access", "scale-up", "startup growth", "unicorns",
        "product to market", "late-stage funding", "early-stage funding", 
        "patent", "patenting", "intellectual property", "ip", "pct" 
        ],
    "competitiveness_comparison": [
        "comparison", "benchmark", "vs", "versus", "global leaders", "oecd",
        "usa", "united states", "singapore", "switzerland", "uk", "eu",
        "china", "japan", "korea", "israel", "germany", "france"
        ],
    "specific_sectors": [
        "ai", "artificial intelligence", "biotech", "biotechnology", "clean tech",
        "quantum computing", "semiconductor", "chips act", "fintech", "digital",
        "advanced manufacturing", "deep-tech", "high-tech"
        ],
     "energy_infrastructure": [
         "energy", "electricity", "grid", "infrastructure", "power quality",
         "data center", "consumption", "hydropower", "nuclear", "kwh",
         "seiia", "regional energy hubs", "grid modernization"
         ]
    }

# Initialize PhraseMatcher for efficient topic matching
matcher = PhraseMatcher(nlp.vocab)
for topic, keywords in topics.items():
    patterns = [nlp(text) for text in keywords]
    matcher.add(topic, patterns)

# Define patterns for extracting references (e.g., "Figure 1", "Table 2")
reference_patterns = [
    r'Figure \d+',
    r'Table \d+',
    r'Appendix [A-Z]\d+',
]
reference_regex = '|'.join(reference_patterns)

# --- File Extraction Functions ---

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return [(text, "text")], os.path.basename(file_path), "report"

def extract_text_from_ipynb(file_path):
    """Extract text, code, and table outputs from a Jupyter notebook."""
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = read_notebook(f, as_version=4)
    contents = []
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            contents.append((''.join(cell.source), "text"))
        elif cell.cell_type == 'code':
            contents.append((''.join(cell.source), "code"))
            if 'outputs' in cell:
                for output in cell.outputs:
                    if output.output_type in ['execute_result', 'display_data']:
                        if 'data' in output and 'text/plain' in output.data:
                            table_text = ''.join(output.data['text/plain'])
                            contents.append((table_text, "table"))
    return contents, os.path.basename(file_path), "notebook"

def extract_text_from_txt(file_path):
    """Extract text from a plain text or markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return [(text, "text")], os.path.basename(file_path), "text"

def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from a Word document.
    """
    text = ""

    doc = Document(file_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return [(text, "text")], os.path.basename(file_path), "report"

def extract_text_from_file(file_path):
    """Dispatch file extraction based on file type."""
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.ipynb'):
        return extract_text_from_ipynb(file_path)
    elif file_path.endswith('.txt') or file_path.endswith('.md'):
        return extract_text_from_txt(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        print(f"Unsupported file type: {file_path}")
        return None

# --- Chunking Function ---

def chunk_text(text, chunk_size=900, overlap=100):
    """Split text into semantically coherent chunks with overlap."""
    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    units = []
    for para in paragraphs:
        if len(para) > 2000:  # If paragraph is too long, split into sentences
            doc = nlp(para)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            units.extend(sentences)
        else:
            units.append(para)
    
    # Group units into chunks
    chunks = []
    current_chunk = []
    current_length = 0
    for unit in units:
        unit_length = len(unit)
        if current_length + unit_length > chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(unit)
        current_length += unit_length
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    # Add overlap between chunks
    if len(chunks) > 1:
        final_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            if len(prev_chunk) > overlap:
                overlap_start = prev_chunk.rfind(' ', 0, -overlap)
                if overlap_start == -1:
                    overlap_start = len(prev_chunk) - overlap
                overlap_text = prev_chunk[overlap_start:]
            else:
                overlap_text = prev_chunk
            current_chunk = overlap_text + '\n' + chunks[i]
            final_chunks.append(current_chunk)
        return final_chunks
    elif chunks:  # If there's exactly one chunk
        return [chunks[0]]
    else:  # If there are no chunks
        return []

# --- Context Generation Function ---

def generate_context(files):
    """Generate structured context from a list of files with unique chunk IDs."""
    context_data = []
    
    for file_path in files:
        result = extract_text_from_file(file_path)
        if not result:
            continue
        
        contents, source_name, source_type = result
        
        # Initialize indices for each content type for this file
        type_indices = {"text": 0, "code": 0, "table": 0}
        
        for content, content_type in contents:
            if content_type == "table":
                chunks = [content]
            else:
                chunks = chunk_text(content)
            
            for chunk in chunks:
                chunk_id = f"{source_name}_{content_type}_{type_indices[content_type]}"
                type_indices[content_type] += 1
                
                # Process chunk with spaCy
                doc = nlp(chunk)
                
                # Extract topics using PhraseMatcher
                matches = matcher(doc)
                topic_counts = {}
                for match_id, start, end in matches:
                    topic = nlp.vocab.strings[match_id]
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                num_tokens = len(doc) or 1  # Avoid division by zero
                topics_list = [
                    {"name": topic, "confidence": count / num_tokens}
                    for topic, count in topic_counts.items()
                    if count / num_tokens > 0
                ]
                topics_list.sort(key=lambda x: x["confidence"], reverse=True)
                
                # Extract entities using spaCy's NER
                entities = [ent.text for ent in doc.ents]
                
                # Extract references (e.g., "Figure 1", "Appendix A1")
                references = list(set(re.findall(reference_regex, chunk)))
                
                # Construct metadata
                metadata = {
                    "source": source_name,
                    "source_type": source_type,
                    "content_type": content_type,
                    "topics": topics_list,
                    "entities": entities,
                    "references": references
                }
                
                # Add to context data
                context_data.append({
                    "id": chunk_id,
                    "content": chunk,
                    "metadata": metadata
                })
    
    return context_data

# --- Example Usage ---

if __name__ == "__main__":
    # List of files to process
    input_dir = "./input_documents"
    os.makedirs(input_dir, exist_ok=True)
    
    # Get all files in input directory
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
             if os.path.isfile(os.path.join(input_dir, f))]
    

    
    # Generate context
    context_data = generate_context(files)
    
    # Save to JSON file
    with open("./data/generated_context_new.json", "w", encoding="utf-8") as f:
        json.dump(context_data, f, ensure_ascii=False, indent=4)
    
    print("Context generated and saved to 'generated_context_new.json'")