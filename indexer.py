import os
import json
from pathlib import Path
import base64
import anthropic
from openai import OpenAI
import PyPDF2
from docx import Document
from PIL import Image
import io


def get_api_key(api_name):
    env_var = f"{api_name.upper()}_API_KEY"

    api_key = input(f"Please enter your {api_name} API key: ").strip()

    if api_key:
        os.environ[env_var] = api_key
        print(f"{api_name} API key set successfully.")
    else:
        print(f"No {api_name} API key provided. Exiting.")
        exit(1)

    return api_key


def summarize_document_anthropic(file_path, client):
    print(f"Summarizing with Anthropic: {file_path}")
    file_content = read_file_content(file_path)
    if file_content is None:
        return None

    system_message = """
    The assistant's job is to summarize the given article into 3-4 sentences. The first sentence should be the overview of the file, and the rest should be the main points of the article. The summary's language should be the same as the passage use.
    Here is the format for the summary:
    \"\"\"
    This file is about .... The main points are: {{first phrase}}, {{second phrase}}, {{third phrase}}, ...
    \"\"\"
    or
    \"\"\"
    此文件是关于... 主要内容是：{{第一个关键词（组）}}，{{第二个关键词（组）}}，{{第三个关键词（组）}}，...
    \"\"\"
    """

    try:
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1500,
            temperature=0.3,
            system=system_message,
            messages=[
                {
                    "role": "user",
                    "content": f"File name: {file_path.name}\n\nFile content:\n{file_content[:2000]}"
                    # Limit content to 2000 characters
                }
            ]
        )
        print(f"Summary generated for: {file_path}")
        return message.content[0].text if message.content else None
    except anthropic.APIError as e:
        print(f"API error occurred: {e}")
        return None


def summarize_document_openai(file_path, client):
    print(f"Summarizing with OpenAI: {file_path}")
    file_content = read_file_content(file_path)
    if file_content is None:
        return None

    system_message = """
    The assistant's job is to summarize the given article into 3-4 sentences. The first sentence should be the overview of the file, and the rest should be the main points of the article. The summary's language should be the same as the passage use.
    Here is the format for the summary:
    \"\"\"
    This file is about .... The main points are: {{first phrase}}, {{second phrase}}, {{third phrase}}, ...
    \"\"\"
    or
    \"\"\"
    此文件是关于... 主要内容是：{{第一个关键词（组）}}，{{第二个关键词（组）}}，{{第三个关键词（组）}}，...
    \"\"\"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"File name: {file_path.name}\n\nFile content:\n{file_content[:2000]}"}
                # Limit content to 2000 characters
            ],
            max_tokens=1000,
            temperature=0.5
        )
        print(f"Summary generated for: {file_path}")
        return response.choices[0].message.content
    except Exception as e:
        print(f"API error occurred: {e}")
        return None

def get_image_media_type(file_path):
    try:
        with Image.open(file_path) as img:
            format = img.format.lower()
            if format == 'jpeg':
                return 'image/jpeg'
            elif format == 'png':
                return 'image/png'
            elif format == 'gif':
                return 'image/gif'
            elif format == 'webp':
                return 'image/webp'
            else:
                raise ValueError(f"Unsupported image type: {format}")
    except Exception as e:
        raise ValueError(f"Error determining image type: {e}")

def summarize_image_anthropic(file_path, client):
    print(f"Summarizing image with Anthropic: {file_path}")

    try:
        media_type = get_image_media_type(file_path)
        try:
            with open(file_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error reading image file {file_path}: {e}")
            return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

    system_message = """
    The assistant's job is to summarize the given image into 3-4 sentences. The first sentence should be an overview, and the rest should describe the main elements or features of the image. And if the image contains text, please include the text in the summary.
    Here is the format for the summary:
    \"\"\"
    This image is about .... The main points are: {{first phrase}}, {{second phrase}}, {{third phrase}}, ...
    \"\"\"
    or
    \"\"\"
    此图片是关于... 主要内容是：{{第一个关键词（组）}}，{{第二个关键词（组）}}，{{第三个关键词（组）}}，...
    \"\"\"
    """

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1500,
            temperature=0.3,
            system=system_message,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            }
                        },
                        {"type": "text",
                         "text": "Here is the image to summarize:"}
                    ]
                }
            ]
        )
        print(f"Summary generated for image: {file_path}")
        return message.content[0].text if message.content else None
    except anthropic.APIError as e:
        print(f"API error occurred: {e}")
        return None


def summarize_image_openai(file_path, client):
    print(f"Summarizing image with OpenAI: {file_path}")

    try:
        media_type = get_image_media_type(file_path)
        try:
            with open(file_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error reading image file {file_path}: {e}")
            return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

    system_message = """
    The assistant's job is to summarize the given image into 3-4 sentences. The first sentence should be an overview, and the rest should describe the main elements or features of the image. And if the image contains text, please include the text in the summary.
    Here is the format for the summary:
    \"\"\"
    This image is about .... The main points are: {{first phrase}}, {{second phrase}}, {{third phrase}}, ...
    \"\"\"
    or
    \"\"\"
    此图片是关于... 主要内容是：{{第一个关键词（组）}}，{{第二个关键词（组）}}，{{第三个关键词（组）}}，...
    \"\"\"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "Summarize this image in 3-4 sentences. The first sentence should be an overview, and the rest should describe the main elements or features of the image."},
                        {
                            "type": "image_url",
                            "image_url": f"data:{media_type};base64,{image_data}",
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        print(f"Summary generated for image: {file_path}")
        return response.choices[0].message.content
    except Exception as e:
        print(f"API error occurred: {e}")
        return None

def read_file_content(file_path):
    suffix = file_path.suffix.lower()
    try:
        if suffix in ['.jpg', '.jpeg', '.png', '.gif']:
            return "<<image_file>>"
        elif suffix == '.pdf':
            return read_pdf(file_path)
        elif suffix == '.docx':
            return read_docx(file_path)
        elif suffix in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            print(f"Unsupported file type: {suffix}")
            return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def read_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages[:5]:  # Limit to first 5 pages
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        return None

def read_docx(file_path):
    try:
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX file {file_path}: {e}")
        return None


def index_folder(folder_path, summarize_document, summarize_image):
    folder_overview = []

    print(f"Indexing folder: {folder_path}")
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = Path(root) / file
            relative_path = file_path.relative_to(folder_path)
            suffix = file_path.suffix.lower()

            if suffix in ['.jpg', '.jpeg', '.png', '.gif']:
                summary = summarize_image(file_path)
            elif suffix in ['.txt', '.md', '.pdf', '.docx']:
                summary = summarize_document(file_path)
            else:
                continue

            if summary:
                folder_overview.append({
                    'file_id': str(relative_path),
                    'file_name': file,
                    'file_path': str(relative_path),
                    'summary': summary
                })
            else:
                print(f"Failed to summarize {file_path}")

    return folder_overview


def main():
    print("Welcome to the Document and Image Indexer and Summarizer!")
    print("This script supports both Anthropic and OpenAI models.")

    while True:
        model_choice = input("Enter 'a' for Anthropic or 'o' for OpenAI: ").lower()

        if model_choice == 'a':
            api_key = get_api_key('anthropic')
            client = anthropic.Anthropic(api_key=api_key)
            summarize_document = lambda file_path: summarize_document_anthropic(file_path, client)
            summarize_image = lambda file_path: summarize_image_anthropic(file_path, client)
            break
        elif model_choice == 'o':
            api_key = get_api_key('openai')
            client = OpenAI(api_key=api_key)
            summarize_document = lambda file_path: summarize_document_openai(file_path, client)
            summarize_image = lambda file_path: summarize_image_openai(file_path, client)
            break
        else:
            print("Invalid choice. Please enter 'a' or 'o'.")

    folder_path = input("Enter the folder path to index: ")
    folder_path = Path(folder_path).resolve()

    if not folder_path.is_dir():
        print("Invalid folder path.")
        return

    print(f"Starting to index folder: {folder_path}")
    folder_overview = index_folder(folder_path, summarize_document, summarize_image)

    if folder_overview:
        output_file = folder_path / 'folder_overview.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(folder_overview, f, ensure_ascii=False, indent=2)
        print(f"Folder overview has been saved to {output_file}")
    else:
        print("No documents or images were successfully summarized.")

if __name__ == "__main__":
    main()