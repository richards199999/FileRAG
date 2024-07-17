# FileRAG: File-based Retrieval-Augmented Generation System

## Motivation

Traditional Retrieval-Augmented Generation (RAG) systems often struggle with maintaining context and coherence when dealing with large documents or complex information structures. FileRAG was born out of the need for a more efficient and context-aware document retrieval system.

The primary motivations for developing FileRAG are:

1. **Preserving Document Context**: Unlike traditional RAG systems that often retrieve fragmented text snippets, FileRAG maintains the integrity of entire documents, ensuring that the context and coherence of information are preserved.

2. **Improved Precision**: By summarizing and indexing entire documents, FileRAG achieves higher precision in retrieving relevant information, especially crucial in fields such as academia, legal research, and technical documentation.

3. **Scalability**: The file-based approach allows for easier management and updating of the knowledge base, making it more scalable for growing document collections.

By addressing these challenges, FileRAG aims to provide a more robust and effective solution for knowledge/document/information retrieval and information extraction tasks.

## Features

- **Dual Model Support**: Choose between Anthropic's Claude and OpenAI's GPT-4 for document summarization and retrieval.
- **Multiple File Format Support**: Handles PDF, DOCX, TXT, and MD files.
- **Intelligent Summarization**: Generates concise summaries of documents for efficient indexing.
- **Context-Aware Retrieval**: Retrieves relevant documents based on user queries using advanced language models.
- **Flexible API Integration**: Easily switch between different AI providers (Anthropic and OpenAI).

## Components

1. **Document Indexer** (`indexer.py`): Indexes and summarizes documents in a specified folder.
2. **Document Retriever** (`retriever.py`): Retrieves relevant documents based on user queries.

## Prerequisites

- Python 3.6+
- `anthropic` library
- `openai` library
- `PyPDF2` library
- `python-docx` library

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/FileRAG.git
   cd FileRAG
   ```

2. Install the required dependencies:
   ```
   pip install anthropic openai PyPDF2 python-docx
   ```

## Usage

### Document Indexer

1. Run the indexer:
   ```
   python document_indexer.py
   ```

2. Choose the AI model (Anthropic or OpenAI) when prompted.

3. Enter your API key for the chosen provider.

4. Specify the folder path containing the documents you want to index.

5. The script will generate a `folder_overview.json` file in the specified folder.

### Document Retriever

1. Run the retriever:
   ```
   python document_retriever.py
   ```

2. Choose the AI model (Anthropic or OpenAI) when prompted.

3. Enter your API key for the chosen provider.

4. Specify the path to the `folder_overview.json` file created by the indexer.

5. Enter your queries when prompted. The script will retrieve relevant documents and save them to `retrieve_result.txt`.

## Configuration

- API keys can be set as environment variables (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) or entered when prompted.
- Adjust the `max_tokens` and `temperature` parameters in the API calls to fine-tune the model outputs.

## Limitations

- The system currently processes only the first 5 pages of PDF documents to manage processing time and API usage.
- Large files may be truncated to fit within API token limits.

## Contributing

Contributions to FileRAG are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgement

Great thanks to Claude-3.5 Sonnet from Anthropic for bringing the idea come to life together!
