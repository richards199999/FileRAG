# FileRAG: File-based Multimodal Retrieval-Augmented Generation System

## Motivation

Traditional Retrieval-Augmented Generation (RAG) systems often struggle with maintaining context and coherence when dealing with large documents or complex information structures. FileRAG was born out of the need for a more efficient and context-aware knowledge/document/information retrieval system.

The primary motivations for developing FileRAG are:

1. **Preserving Document Context**: Unlike traditional RAG systems that often retrieve fragmented text snippets, FileRAG maintains the integrity of entire documents, ensuring that the context and coherence of information are preserved.

2. **Multimodal Indexing**: By using frontier models with vision and audio capabilities, FileRAG is able to index text, images, and audio files, providing a comprehensive file retrieval system.

3. **Improved Precision**: By summarizing and indexing entire documents, FileRAG achieves higher precision in retrieving relevant information, especially crucial in fields such as academia, legal research, technical documentation, and more.

4. **Scalability**: The file-based approach allows for easier management and updating of the knowledge base, making it more scalable for growing document collections.

By addressing these challenges, FileRAG aims to provide a more robust and effective solution for knowledge/document/information retrieval and information extraction tasks.

## Features

- **Dual Model Support**: Choose between Anthropic's Claude and OpenAI's GPT-4 for document summarization and retrieval.
- **Multiple File Format Support**: Handles PDF, DOCX, TXT, MD, various image files (JPEG, PNG, GIF, WEBP), and audio files (MP3, WAV, OGG, FLAC, AAC, OPUS, M4A, MP4, MPEG, MOV, WEBM).
- **Intelligent Summarization**: Generates concise summaries of files for efficient indexing, including specialized summarization for audio transcripts.
- **Context-Aware Retrieval**: Retrieves relevant files based on user queries using advanced language models.
- **Flexible API Integration**: Easily switch between different AI providers (Anthropic and OpenAI) for summarization and audio transcription (OpenAI and Lemonfox.ai).
- **Organized Results**: Stores retrieval results in a structured folder system, separating text, image, and audio results for easy access and review.

## Structure

![image](https://github.com/user-attachments/assets/ccc56f7a-e613-4a45-8426-59c1be6c0109)

## Components

1. **File Indexer** (`indexer.py`): Indexes and summarizes files in a specified folder, including text, images, and audio.
2. **File Retriever** (`retriever.py`): Retrieves relevant files based on user queries, handling text, images, and audio files.

## Prerequisites

- Python 3.6+
- `anthropic` library
- `openai` library
- `PyPDF2` library
- `python-docx` library
- `Pillow` library

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/richards199999/FileRAG.git
   cd FileRAG
   ```

2. Install the required dependencies:
   ```
   pip install anthropic openai PyPDF2 python-docx Pillow
   ```

## Usage

### File Indexer

1. Run the indexer:
   ```
   python indexer.py
   ```

2. Choose the AI model (Anthropic or OpenAI) for summarization when prompted.

3. Choose the API (OpenAI or Lemonfox.ai) for audio transcription when prompted.

4. Enter your API key(s) for the chosen provider(s).

5. Specify the folder path containing the documents, images, and audio files you want to index.

6. The script will generate a `folder_overview.json` file in the specified folder.

### File Retriever

1. Run the retriever:
   ```
   python retriever.py
   ```

2. Choose the AI model (Anthropic or OpenAI) when prompted.

3. Enter your API key for the chosen provider.

4. Specify the path to the `folder_overview.json` file created by the indexer.

5. Enter your queries when prompted. The script will retrieve relevant documents, images, and audio files, saving them in the `filerag_results` folder:
   - Text results are saved in `text_results/YYYYMMDD_HHMMSS/retrieved_text_results.txt`
   - Image results are copied to `image_results/YYYYMMDD_HHMMSS/`
   - Audio results are copied to `audio_results/YYYYMMDD_HHMMSS/`

## Configuration

- API keys can be set as environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `LEMONFOX_API_KEY`) or entered when prompted.
- Adjust the `max_tokens` and `temperature` parameters in the API calls to fine-tune the model outputs.

## Limitations

- The system currently processes only the first 5 pages of PDF documents to manage processing time and API usage.
- Large files may be truncated to fit within API token limits.
- Image and audio processing capabilities depend on the chosen AI model's features.

## Contributing

Contributions to FileRAG are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgement

Great thanks to Claude-3.5 Sonnet from Anthropic for bringing the idea to life together!🤗