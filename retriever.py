import json
from pathlib import Path
import anthropic
from openai import OpenAI
import os
import datetime
import docx
import PyPDF2
import re


def get_api_key(api_name):
    env_var = f"{api_name.upper()}_API_KEY"

    # Always prompt for the API key
    api_key = input(f"Please enter your {api_name} API key: ").strip()

    if api_key:
        os.environ[env_var] = api_key
        print(f"{api_name} API key set successfully.")
    else:
        print(f"No {api_name} API key provided. Exiting.")
        exit(1)

    return api_key


def load_folder_overview(overview_path):
    print(f"Loading folder overview from {overview_path}")
    with open(overview_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from folder overview")
    return data


def log_api_response(response, query, log_file):
    print(f"Logging API response to {log_file}")
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

    file_id_tuples = re.findall(r'"([^"]+\.(md|pdf|txt|docx))"', response_content)

    file_ids = [Path(file_id[0]).name for file_id in file_id_tuples]

    print(f"Parsed and normalized file IDs: {file_ids}")
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
    The assistant's job is to pick the best file(s) that match(es) the given query using the file name and the file summary. If there is multiple file, use comma to seperate. It only need to return the file id in a JSON format. It should not miss any file that is related to the query. It *MUST* only return the JSON string without any other text, or it will be considered as an error.
    Here is the query:
    \"\"\"
    {query}
    \"\"\"
    Here is the overview of the knowledge base's structure:
    \"\"\"
    {folder_overview}
    \"\"\"
    Example output format (It must follow this format, or it will be considered as an error):
    \"\"\"
    {
        "file_id": "file_id1", "file_id2", ...
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
    The assistant's job is to pick the best file(s) that match(es) the given query using the file name and the file summary. If there is multiple file, use comma to seperate. It only need to return the file id in a JSON format. It should not miss any file that is related to the query. It *MUST* only return the JSON string without any other text, or it will be considered as an error.
    Here is the query:
    \"\"\"
    {query}
    \"\"\"
    Here is the overview of the knowledge base's structure:
    \"\"\"
    {folder_overview}
    \"\"\"
    Example output format (It must follow this format, or it will be considered as an error):
    \"\"\"
    {
        "file_id": "file_id1", "file_id2", ...
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
                if full_path.suffix.lower() == '.pdf':
                    content = extract_pdf_content(full_path)
                    print(f"PDF file content retrieved: {full_path}")
                elif full_path.suffix.lower() == '.docx':
                    content = extract_docx_content(full_path)
                    print(f"Word file content retrieved: {full_path}")
                else:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    print(f"Text file content retrieved: {full_path}")
                return item['file_path'], content
            except Exception as e:
                print(f"Error reading file {full_path}: {e}")
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


def write_results(results, output_file):
    print(f"Writing results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (file_path, content) in enumerate(results, 1):
            f.write(f'--- Retrieved Document {i} ---\n')
            f.write(f'Original File Path: "{file_path}"\n')
            f.write('Original File Content:\n')
            if content == "<<Non-text file>>":
                f.write(content + "\n")
            else:
                f.write('"""\n')
                f.write(content)
                f.write('\n"""\n')
            f.write('\n')
    print(f"Results written to {output_file}")

    # Verify the file was written
    if os.path.exists(output_file):
        print(f"Verified: {output_file} exists.")
        print(f"File size: {os.path.getsize(output_file)} bytes")
    else:
        print(f"Error: {output_file} was not created.")


def main():
    print("Welcome to the Document Retriever!")
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
    log_file = folder_path / 'api_response_log.txt'

    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        file_ids = process_query(query, folder_overview, client, log_file)
        print(f"File IDs returned by process_query: {file_ids}")
        if file_ids:
            results = []
            for file_id in file_ids:
                retrieved_path, content = retrieve_document(file_id, folder_path, folder_overview)
                if retrieved_path:
                    results.append((retrieved_path, content))
                    print(f"Retrieved document: {retrieved_path}")
                else:
                    print(f"Error: Unable to retrieve the document with file ID: {file_id}")

            if results:
                output_file = folder_path / 'retrieve_result.txt'
                write_results(results, output_file)
                print(f"Retrieved content has been saved to {output_file}")
            else:
                print("No documents could be retrieved.")
        else:
            print("No matching documents found.")

    print(f"API response log has been saved to {log_file}")
    print("Document retrieval process completed.")


if __name__ == "__main__":
    main()
