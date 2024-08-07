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
import cv2
import numpy as np


def get_api_key(api_name):
    env_var = f"{api_name.upper()}_API_KEY"

    api_key = os.getenv(env_var)

    if not api_key:
        api_key = input(f"Please enter your {api_name} API key: ").strip()

        if api_key:
            os.environ[env_var] = api_key
            print(f"{api_name} API key set successfully.")
        else:
            print(f"No {api_name} API key provided. Exiting.")
            exit(1)

    else:
        print(f"Using {api_name} API key from environment.")

    return api_key


def summarize_document(file_path, client):
    print(f"Summarizing document: {file_path}")
    file_content = read_file_content(file_path)
    if file_content is None:
        return None

    system_message = """
    The assistant's job is to summarize the given article into 3-4 sentences. The first sentence should be the overview of the file, and the rest should be the main points of the article. The summary's language must be the same as the passage use.
    Here is the format for the summary:
    \"\"\"
    This file is about .... The main points are: {{first phrase}}, {{second phrase}}, {{third phrase}}, ...
    \"\"\"
    """

    try:
        if isinstance(client, anthropic.Anthropic):
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1500,
                temperature=0.3,
                system=system_message,
                messages=[
                    {
                        "role": "user",
                        "content": f"File name: {file_path.name}\n\nFile content:\n{file_content[:2000]}"
                    }
                ]
            )
            summary = message.content[0].text if message.content else None
        else:  # OpenAI
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"File name: {file_path.name}\n\nFile content:\n{file_content[:2000]}"}
                ],
                max_tokens=1000,
                temperature=0.5
            )
            summary = response.choices[0].message.content
        
        print(f"Summary generated for: {file_path}")
        return summary
    except Exception as e:
        print(f"API error occurred: {e}")
        return None


def summarize_image(file_path, client):
    print(f"Summarizing image: {file_path}")

    try:
        media_type = get_image_media_type(file_path)
        with open(file_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error processing image file {file_path}: {e}")
        return None

    system_message = """
    The assistant's job is to summarize the given image into 3-4 sentences. The first sentence should be an overview, and the rest should describe the main elements or features of the image. And if the image contains text, please include the text in the summary.
    Here is the format for the summary:
    \"\"\"
    This image is about .... The main points are: {{first phrase}}, {{second phrase}}, {{third phrase}}, ...
    \"\"\"
    """

    try:
        if isinstance(client, anthropic.Anthropic):
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
                            {"type": "text", "text": "Here is the image to summarize:"}
                        ]
                    }
                ]
            )
            summary = message.content[0].text if message.content else None
        else:  # OpenAI
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Summarize this image in 3-4 sentences."},
                            {
                                "type": "image_url",
                                "image_url": f"data:{media_type};base64,{image_data}",
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            summary = response.choices[0].message.content
        
        print(f"Summary generated for image: {file_path}")
        return summary
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


def transcribe_audio_openai(file_path, client):
    print(f"Transcribing audio with OpenAI: {file_path}")
    try:
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        print(f"Error transcribing audio file {file_path}: {e}")
        return None


def transcribe_audio_lemonfox(file_path, client):
    print(f"Transcribing audio with Lemonfox: {file_path}")
    try:
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        print(f"Error transcribing audio file {file_path}: {e}")
        return None


def summarize_audio_transcript(transcript, client):
    print("Summarizing audio transcript")
    system_message = """
    The assistant's job is to summarize the given audio transcription into 3-4 sentences. The first sentence should be the overview of the audio, and the rest should be the main points of the audio. The summary's language must be the same as the original audio use.

    Here is the format for the summary:
    \"\"\"
    This audio is about .... The main points are: {{first phrase}}, {{second phrase}}, {{third phrase}}, ...
    \"\"\"
    """

    try:
        if isinstance(client, anthropic.Anthropic):
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                temperature=0.3,
                system=system_message,
                messages=[
                    {
                        "role": "user",
                        "content": f"Audio transcript:\n{transcript[:2000]}"
                    }
                ]
            )
            summary = message.content[0].text if message.content else None
        else:  # OpenAI
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Audio transcript:\n{transcript[:2000]}"}
                ],
                max_tokens=2000,
                temperature=0.5
            )
            summary = response.choices[0].message.content
        
        print("Audio summary generated")
        return summary
    except Exception as e:
        print(f"API error occurred: {e}")
        return None


def summarize_audio(file_path, summarization_client, transcription_client, transcribe_function):
    transcript = transcribe_function(file_path, transcription_client)
    if transcript:
        return summarize_audio_transcript(transcript, summarization_client)
    return None


def extract_key_frames(video_path, num_frames=5):
    video_path_str = str(video_path)  # Convert Path to string
    video = cv2.VideoCapture(video_path_str)
    if not video.isOpened():
        print(f"Error opening video file: {video_path_str}")
        return []

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    key_frames = []
    for idx in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video.read()
        if ret:
            key_frames.append(frame)

    video.release()
    return key_frames


def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def summarize_video_frames(frames, client):
    print("Summarizing video frames")
    encoded_frames = [encode_frame(frame) for frame in frames]

    system_message = """
    The assistant's job is to summarize the given video frames into 3-4 sentences. The first sentence should be an overview, and the rest should describe the main elements or features of the frame. And if the frame contains text, please include the text in the summary.

    Here is the format for the summary:
    \"\"\"
    This video is about .... The main points are: {{first phrase}}, {{second phrase}}, {{third phrase}}, ...
    \"\"\"
    """

    try:
        if isinstance(client, anthropic.Anthropic):
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1500,
                temperature=0.3,
                system=system_message,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Here are the video frames."},
                            *[{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": frame}} for frame in encoded_frames]
                        ]
                    }
                ]
            )
            summary = message.content[0].text if message.content else None
        else:  # OpenAI
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Here are the video frames."},
                            *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}} for frame in encoded_frames]
                        ]
                    }
                ],
                max_tokens=300,
            )
            summary = response.choices[0].message.content
        
        print("Key frames summary generated")
        return summary
    except Exception as e:
        print(f"API error occurred: {e}")
        return None


def summarize_video(file_path, summarization_client, transcription_client, transcribe_function):
    print("Understanding video")
    key_frames = extract_key_frames(file_path)
    if not key_frames:
        print(f"Failed to extract key frames from {file_path}")
        return None

    frames_summary = summarize_video_frames(key_frames, summarization_client)

    audio_summary = summarize_audio(file_path, summarization_client, transcription_client, transcribe_function)

    system_message = """
    The assistant's job is to summarize the given video into 3-4 sentences by using the description of the frames and the background audio. The first sentence should be an overview, and the rest should describe the main elements or features of the video. And if the frame contains text, please include the text in the summary.

    Here is the format for the summary:
    \"\"\"
    This video is about .... The main points are: {{first phrase}}, {{second phrase}}, {{third phrase}}, ...
    \"\"\"
    """

    try:
        if isinstance(summarization_client, anthropic.Anthropic):
            message = summarization_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1500,
                temperature=0.3,
                system=system_message,
                messages=[
                    {
                        "role": "user",
                        "content": f"Key frames summary: {frames_summary}\nBackground audio summary: {audio_summary}"
                    }
                ]
            )
            print("Video summary generated")
            return message.content[0].text if message.content else None
        else:
            response = summarization_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user",
                     "content": f"Key frames summary: {frames_summary}\nBackground audio summary: {audio_summary}"}
                ],
                max_tokens=300,
            )
            print("Video summary generated")
            return response.choices[0].message.content
    except Exception as e:
        print(f"API error occurred: {e}")
        return None


def index_folder(folder_path, summarize_document, summarize_image, summarize_audio, summarize_video):
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
            elif suffix in ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.opus', '.m4a']:
                summary = summarize_audio(file_path)
            elif suffix in ['.mp4', '.avi', '.mov', '.mkv']:
                summary = summarize_video(file_path)
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
    print("Welcome to the Multimodal File Indexer!")
    print("This script supports both Anthropic and OpenAI models for summarization.")
    print("For audio transcription, you can choose between OpenAI and Lemonfox.ai.")

    while True:
        model_choice = input("Enter 'a' for Anthropic or 'o' for OpenAI for summarization: ").lower()

        if model_choice == 'a':
            api_key = get_api_key('anthropic')
            summarization_client = anthropic.Anthropic(api_key=api_key)
            break
        elif model_choice == 'o':
            api_key = get_api_key('openai')
            summarization_client = OpenAI(api_key=api_key)
            break
        else:
            print("Invalid choice. Please enter 'a' or 'o'.")

    while True:
        audio_api_choice = input("Enter 'o' for OpenAI or 'l' for Lemonfox.ai for audio transcription: ").lower()

        if audio_api_choice == 'o':
            if model_choice == 'o':
                transcription_client = summarization_client
            else:
                transcription_client = OpenAI(api_key=get_api_key('openai'))
            transcribe_function = transcribe_audio_openai
            print("OpenAI's Audio API will be used for transcription (Whisper-V2).")
            break
        elif audio_api_choice == 'l':
            transcription_client = OpenAI(
                api_key=get_api_key('lemonfox'),
                base_url="https://api.lemonfox.ai/v1",
            )
            transcribe_function = transcribe_audio_lemonfox
            print("Lemonfox.ai will be used for transcription (Whisper-V3).")
            break
        else:
            print("Invalid choice. Please enter 'o' or 'l'.")

    summarize_document_lambda = lambda file_path: summarize_document(file_path, summarization_client)
    summarize_image_lambda = lambda file_path: summarize_image(file_path, summarization_client)
    summarize_audio_lambda = lambda file_path: summarize_audio(file_path, summarization_client, transcription_client, transcribe_function)
    summarize_video_lambda = lambda file_path: summarize_video(file_path, summarization_client, transcription_client, transcribe_function)

    folder_path = input("Enter the folder path to index: ")
    folder_path = Path(folder_path).resolve()

    if not folder_path.is_dir():
        print("Invalid folder path.")
        return

    print(f"Starting to index folder: {folder_path}")
    folder_overview = index_folder(folder_path, summarize_document_lambda, summarize_image_lambda, summarize_audio_lambda, summarize_video_lambda)

    if folder_overview:
        output_file = folder_path / 'folder_overview.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(folder_overview, f, ensure_ascii=False, indent=2)
        print(f"Folder overview has been saved to {output_file}")
    else:
        print("No documents, images, audio files, or videos were successfully summarized.")


if __name__ == "__main__":
    main()
