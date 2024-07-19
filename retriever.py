import base64
import json
import shutil
from pathlib import Path
import anthropic
import cv2
from openai import OpenAI
import os
import datetime
import docx
import PyPDF2
import re


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


def load_folder_overview(overview_path):
    print(f"Loading folder overview from {overview_path}")
    with open(overview_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from folder overview")
    return data


def create_results_folders(base_folder):
    filerag_results = base_folder / 'filerag_results'
    filerag_results.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder = filerag_results / timestamp
    session_folder.mkdir(exist_ok=True)

    image_results = session_folder / 'image_results'
    image_results.mkdir(exist_ok=True)

    text_results = session_folder / 'text_results'
    text_results.mkdir(exist_ok=True)

    audio_results = session_folder / 'audio_results'
    audio_results.mkdir(exist_ok=True)

    video_results = session_folder / 'video_results'
    video_results.mkdir(exist_ok=True)

    return filerag_results, session_folder, image_results, text_results, audio_results, video_results


def log_api_response(response, query, log_file):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n--- API Response Log: {timestamp} ---\n")
        f.write(f"Query: {query}\n")
        f.write(f"Response: {response}\n")
        f.write("-----------------------------------\n")
    print(f"API response logged to {log_file}")


def parse_file_ids(response_content):
    print("Parsing file IDs from API response")
    print(f"Raw response: {response_content}")

    cleaned_response = re.sub(r'```json\s*|\s*```', '', response_content).strip()

    try:
        response_json = json.loads(cleaned_response)

        if 'file_id' in response_json:
            file_ids = response_json['file_id'].split(',')

            normalized_file_ids = [Path(file_id.strip()).name for file_id in file_ids]

            print(f"Parsed and normalized file IDs: {normalized_file_ids}")
            return normalized_file_ids
        else:
            print("No 'file_id' key found in the JSON response")
            return []

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print("Falling back to regex method")
        file_id_tuples = re.findall(r'"([^"]+\.(md|pdf|txt|docx))"', response_content)
        file_ids = [Path(file_id[0]).name for file_id in file_id_tuples]
        print(f"Parsed and normalized file IDs (regex method): {file_ids}")
        return file_ids


def extract_pdf_content(pdf_path, max_pages=1):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        content = ""
        for page in reader.pages[:max_pages]:
            content += page.extract_text() + "\n"
    return content


def process_query_anthropic(query, folder_overview, client, log_file):
    system_message = """
    The assistant's job is to pick the best file(s) that match(es) the given query using the file name and the file summary. If there is multiple file, use comma to seperate. It only need to return the file id in a JSON format. It should not miss any file that is related to the query. It *MUST* only return the JSON string without any other text, or it will be considered as an error. Do not put the JSON string inside the triple backticks, or it will be considered as an error.
    Reminder: The description of the audio file might be affected by the original transcription, so it might have recognition errors; DO NOT be strict with the audio file.
    Example output format (It must follow this format, or it will be considered as an error):
    \"\"\"
    {
        "file_id": "file_id1,file_id2,...“
    }
    \"\"\"
    """
    try:
        print("Sending request to Anthropic API")
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0.5,
            system=system_message,
            messages=[
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nFolder overview:\n{json.dumps(folder_overview, ensure_ascii=False, indent=2)}"
                }
            ]
        )

        response_content = message.content[0].text if message.content else None
        print(f"Received API response: {response_content}")
        log_api_response(response_content, query, log_file)

        if response_content:
            return parse_file_ids(response_content)
        else:
            print("Error: Empty response from the API.")
            return []
    except anthropic.APIError as e:
        print(f"API error occurred: {e}")
        log_api_response(str(e), query, log_file)
        return []


def process_query_openai(query, folder_overview, client, log_file):
    system_message = """
    The assistant's job is to pick the best file(s) that match(es) the given query using the file name and the file summary. If there is multiple file, use comma to seperate. It only need to return the file id in a JSON format. It should not miss any file that is related to the query. It *MUST* only return the JSON string without any other text, or it will be considered as an error. Do not put the JSON string inside the triple backticks, or it will be considered as an error.
    Reminder: The description of the audio file might be affected by the original transcription, so it might have recognition errors; DO NOT be strict with the audio file.
    Example output format (It must follow this format, or it will be considered as an error):
    \"\"\"
    {
        "file_id": "file_id1,file_id2,...“
    }
    \"\"\"
    """
    try:
        print("Sending request to OpenAI API")
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",
                 "content": f"Query: {query}\n\nFolder overview:\n{json.dumps(folder_overview, ensure_ascii=False, indent=2)}"}
            ]
        )

        response_content = completion.choices[0].message.content
        print(f"Received API response: {response_content}")
        log_api_response(response_content, query, log_file)

        if response_content:
            return parse_file_ids(response_content)
        else:
            print("Error: Empty response from the API.")
            return []
    except Exception as e:
        print(f"API error occurred: {e}")
        log_api_response(str(e), query, log_file)
        return []


def retrieve_document(file_id, folder_path, folder_overview):
    print(f"Retrieving document: {file_id}")
    for item in folder_overview:
        if file_id in [item['file_id'], item['file_name'], Path(item['file_path']).name]:
            full_path = folder_path / item['file_path']
            try:
                if full_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                    print(f"Image file found: {full_path}")
                    return str(full_path), "<<image_file>>"
                elif full_path.suffix.lower() in ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.opus', '.m4a']:
                    print(f"Audio file found: {full_path}")
                    return str(full_path), "<<audio_file>>"
                elif full_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    print(f"Video file found: {full_path}")
                    return str(full_path), "<<video_file>>"
                elif full_path.suffix.lower() == '.pdf':
                    content = extract_pdf_content(full_path)
                    print(f"PDF file content retrieved: {full_path}")
                elif full_path.suffix.lower() == '.docx':
                    content = extract_docx_content(full_path)
                    print(f"Word file content retrieved: {full_path}")
                else:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    print(f"Text file content retrieved: {full_path}")
                return str(full_path), content
            except Exception as e:
                print(f"Error reading file {full_path}: {e}")
                return str(full_path), f"<<Error reading file: {e}>>"
    print(f"File ID {file_id} not found in folder overview")
    return None, None


def extract_docx_content(docx_path):
    try:
        doc = docx.Document(docx_path)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        return '\n'.join(fullText).strip()
    except Exception as e:
        print(f"Error reading Word file {docx_path}: {e}")
        return "<<Error reading Word file>>"


def write_results(results, session_folder, is_image=False, is_audio=False, is_video=False):
    if is_image or is_audio or is_video:
        result_folder = session_folder / (
            'image_results' if is_image else 'audio_results' if is_audio else 'video_results')
        for i, (file_path, _) in enumerate(results, 1):
            original_file = Path(file_path)
            new_file_name = f"{i}_{original_file.name}"
            shutil.copy2(original_file, result_folder / new_file_name)
        print(f"{'Image' if is_image else 'Audio' if is_audio else 'Video'} results copied to {result_folder}")
    else:
        output_file = session_folder / 'text_results' / 'retrieved_text_results.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (file_path, content) in enumerate(results, 1):
                f.write(f'--- Retrieved Document {i} ---\n')
                f.write(f'Original File Path: "{file_path}"\n')
                f.write('Original File Content:\n')
                f.write('"""\n')
                f.write(content)
                f.write('\n"""\n\n')
        print(f"Text results written to {output_file}")


def extract_video_frame(video_path, frame_number=0):
    video = cv2.VideoCapture(str(video_path))
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()
    video.release()
    if ret:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    return None


def main():
    print("Welcome to the Multimodal File Retriever!")
    print("This script supports both Anthropic and OpenAI models.")

    while True:
        model_choice = input("Enter 'a' for Anthropic or 'o' for OpenAI: ").lower()

        if model_choice == 'a':
            api_key = get_api_key('anthropic')
            client = anthropic.Anthropic(api_key=api_key)
            process_query = process_query_anthropic
            break
        elif model_choice == 'o':
            api_key = get_api_key('openai')
            client = OpenAI(api_key=api_key)
            process_query = process_query_openai
            break
        else:
            print("Invalid choice. Please enter 'a' or 'o'.")

    overview_path = input("Enter the path to folder_overview.json: ")
    overview_path = Path(overview_path).resolve()

    if not overview_path.is_file():
        print("Invalid folder_overview.json path.")
        return

    folder_path = overview_path.parent
    folder_overview = load_folder_overview(overview_path)
    filerag_results, session_folder, image_results_folder, text_results_folder, audio_results_folder, video_results_folder = create_results_folders(
        folder_path)
    log_file = filerag_results / 'api_response_log.txt'

    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        file_ids = process_query(query, folder_overview, client, log_file)
        print(f"File IDs returned by process_query: {file_ids}")
        if file_ids:
            text_results = []
            image_results = []
            audio_results = []
            video_results = []
            for file_id in file_ids:
                retrieved_path, content = retrieve_document(file_id, folder_path, folder_overview)
                if retrieved_path:
                    if content == "<<image_file>>":
                        image_results.append((retrieved_path, content))
                    elif content == "<<audio_file>>":
                        audio_results.append((retrieved_path, content))
                    elif content == "<<video_file>>":
                        video_results.append((retrieved_path, content))
                    else:
                        text_results.append((retrieved_path, content))
                    print(f"Retrieved document: {retrieved_path}")
                else:
                    print(f"Error: Unable to retrieve the document with file ID: {file_id}")

            if text_results:
                write_results(text_results, session_folder)
            if image_results:
                write_results(image_results, session_folder, is_image=True)
            if audio_results:
                write_results(audio_results, session_folder, is_audio=True)
            if video_results:
                write_results(video_results, session_folder, is_video=True)

            if not text_results and not image_results and not audio_results and not video_results:
                print("No documents could be retrieved.")
        else:
            print("No matching documents found.")

    print(f"API response log has been saved to {log_file}")
    print("Document retrieval process completed.")


if __name__ == "__main__":
    main()
