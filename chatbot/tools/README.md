# Tools for Canada Innovation Strategy Chatbot

This directory contains utility tools for the chatbot project.

## Context Generator

The `context_generator.py` script automatically extracts text from various document formats and generates properly formatted context data for the chatbot. This tool makes it easy to update the chatbot's knowledge base with new information.

### Supported File Types

- PDF (`.pdf`)
- Word documents (`.docx`) 
- Jupyter notebooks (`.ipynb`)
- Plain text files (`.txt`)

### Usage

1. Place your documents in the `input_documents` directory
2. Run the context generator:

```bash
python context_generator.py --output_file ../data/generated_context.json
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input_dir` | Directory containing input documents | `./input_documents` |
| `--output_file` | Output JSON file path for the generated context data | `../data/generated_context.json` |
| `--chunk_size` | Target size of each text chunk in characters | `1000` |
| `--overlap` | Number of characters to overlap between chunks | `200` |

### Example Commands

Generate context data with default settings:
```bash
python context_generator.py
```

Specify custom input and output paths:
```bash
python context_generator.py --input_dir /path/to/documents --output_file /path/to/output.json
```

Adjust chunking parameters:
```bash
python context_generator.py --chunk_size 500 --overlap 100
```

### How It Works

1. **Text Extraction**: The tool extracts text from all supported files in the input directory
2. **Text Chunking**: It splits the extracted text into semantically meaningful chunks of appropriate size
3. **Topic Assignment**: It automatically assigns topics to each chunk based on content analysis
4. **Context Generation**: It creates a properly formatted JSON file with the processed chunks

### Requirements

The following Python packages are required:
- PyPDF2 (for PDF files)
- python-docx (for Word documents)
- nltk (for text processing)

These dependencies are already included in the main `requirements.txt` file.

### After Context Generation

After generating the context data, remember to:

1. Re-index the vector database:
```bash
cd ..
python main.py --mode index
```

2. Restart the application to use the updated context 