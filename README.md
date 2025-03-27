# Canada Innovation Strategy Chatbot

An interactive chatbot powered by Google's Gemini API that allows users to have meaningful conversations about Canada's innovation strategy, based on analysis and findings from MIE1624 course project.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Environment Setup](#environment-setup)
  - [Running Locally](#running-locally)
- [Usage](#usage)
- [Deployment](#deployment)
  - [Container Deployment](#container-deployment)
  - [Cloud Deployment](#cloud-deployment)
- [Updating Context](#updating-context)
  - [Using the Context Generator](#using-the-context-generator)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

This project implements a chatbot that can answer questions about Canada's innovation strategy, policies, R&D factors, and recommendations based on the analysis from the MIE1624 course project. The chatbot uses a Retrieval-Augmented Generation (RAG) approach to provide accurate, contextually relevant responses by leveraging Google's Gemini API.

## Architecture

The project follows a modular architecture:

- **Frontend**: Streamlit-based UI providing an intuitive chat interface
- **Backend**: Flask API handling chat queries, context retrieval, and LLM interaction
- **RAG System**: ChromaDB vector database with Gemini embeddings for semantic search
- **LLM Integration**: Google Gemini API for generating conversational responses

## Technology Stack

- **LLM**: Google Gemini (gemini-1.5-flash-latest)
- **Embeddings**: Gemini Embeddings (models/embedding-001)
- **Vector Database**: ChromaDB
- **Backend**: Python, Flask
- **Frontend**: Streamlit
- **Containerization**: Docker

## Setup

### Prerequisites

- Python 3.8+ installed
- Google Gemini API key (obtain from [Google AI Studio](https://makersuite.google.com/app/apikey))
- Git (for cloning the repository)

### Environment Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/canada-innovation-chatbot.git
cd canada-innovation-chatbot
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:

```bash
cp .env.example .env
# Edit .env file to add your Google API key
```

### Running Locally

1. Index the context data:

```bash
python main.py --mode index
```

2. Run the chatbot (combined frontend and direct backend):

```bash
python main.py --mode all
```

Alternatively, you can run components separately:

```bash
# Run only the backend server
python main.py --mode backend

# In another terminal, run the frontend
python main.py --mode frontend
```

## Usage

Once the application is running, open your browser to access the Streamlit interface (usually at `http://localhost:8501`).

Example questions you can ask:
- "What is Canada's R&D spending as a percentage of GDP?"
- "What are the key recommendations for improving Canada's innovation performance?"
- "How do provincial innovation policies vary across Canada?"
- "What are the main challenges for innovative companies in Canada?"

## Deployment

### Container Deployment

Build and run using Docker:

```bash
docker build -t canada-innovation-chatbot .
docker run -p 8501:8501 -e GOOGLE_API_KEY=your_api_key canada-innovation-chatbot
```

### Cloud Deployment

#### Google Cloud Run

1. Build and push container:

```bash
gcloud builds submit --tag gcr.io/your-project/canada-innovation-chatbot
```

2. Deploy to Cloud Run:

```bash
gcloud run deploy canada-innovation-chatbot \
  --image gcr.io/your-project/canada-innovation-chatbot \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY=your_api_key
```

#### AWS/Azure

Similar deployment steps can be followed for AWS ECS/Fargate or Azure Container Instances.

## Updating Context

### Using the Context Generator

The project includes a context generator tool that can automatically extract and process context data from various document formats (PDF, Word, Jupyter notebooks). This makes it easy to update the chatbot's knowledge base with new information.

1. Place your documents in the `tools/input_documents` directory (or specify a different directory with `--input_dir`):
   - PDF files (`.pdf`)
   - Word documents (`.docx`) 
   - Jupyter notebooks (`.ipynb`)

2. Run the context generator:

```bash
cd chatbot
python tools/context_generator.py --output_file data/generated_context.json
```

3. Additional options:
```bash
python tools/context_generator.py --help
```

The tool will:
- Extract text from all supported files
- Split the content into appropriate chunks
- Automatically assign topics based on content
- Generate a properly formatted context JSON file

4. Re-index the vector database with the new context:

```bash
python main.py --mode index
```

5. Restart the application to use the updated context.

## Project Structure

```
chatbot/
├── backend/
│   ├── __init__.py
│   ├── app.py            # Flask API and core chat logic
│   ├── gemini_interface.py  # Gemini API interactions
│   ├── rag_module.py     # Vector DB and retrieval functions
│   └── index_context.py  # Context indexing script
├── data/
│   ├── sample_context.json  # Core knowledge chunks
│   └── chroma_db/        # Vector database files (generated)
├── frontend/
│   └── frontend_streamlit.py  # Streamlit UI
├── tools/
│   ├── context_generator.py  # Tool to generate context from documents
│   └── input_documents/   # Place input files here
├── .env.example          # Example environment variables
├── .gitignore            # Git ignore file
├── Dockerfile            # Container definition
├── main.py               # Main entry point script
├── README.md             # This documentation
└── requirements.txt      # Python dependencies
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
