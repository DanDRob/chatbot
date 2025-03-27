#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Canada Innovation Strategy Chatbot Context Generator ===${NC}"
echo

# Check if the input_documents directory exists and has files
if [ ! -d "input_documents" ] || [ -z "$(ls -A input_documents 2>/dev/null)" ]; then
    echo -e "${YELLOW}Warning: The input_documents directory is empty or doesn't exist.${NC}"
    echo "Please add some documents (PDF, DOCX, IPYNB, TXT) to the input_documents directory."
    echo "Creating the directory if it doesn't exist..."
    mkdir -p input_documents
    echo
    echo "Example: Copy some documents using the commands below:"
    echo "  cp /path/to/your/document.pdf input_documents/"
    echo "  cp /path/to/your/document.docx input_documents/"
    echo "  cp /path/to/your/document.txt input_documents/"
    echo
    exit 1
fi

echo "Found the following files in input_documents:"
ls -l input_documents
echo

# Ask for confirmation before proceeding
read -p "Do you want to generate context data from these files? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

echo -e "${GREEN}Running context generator...${NC}"
echo

# Run the context generator script
python context_generator.py --output_file ../data/generated_context.json

# Check if the operation was successful
if [ $? -eq 0 ]; then
    echo
    echo -e "${GREEN}Context data generated successfully!${NC}"
    echo "Next steps:"
    echo "1. Go back to the main directory: cd .."
    echo "2. Re-index the vector database: python main.py --mode index"
    echo "3. Restart the chatbot: python main.py --mode all"
else
    echo
    echo -e "${YELLOW}Error generating context data. Please check the logs above.${NC}"
fi 