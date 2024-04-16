import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import backoff
import openai
import logging
import re
from urllib.parse import urlparse

# Define a dictionary to store processed documents with timestamps
processed_documents = {}

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

# OpenAI API key
API_KEY = "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"

# GPT-3 API endpoint
GPT3_API_URL = "https://api.openai.com/v1/chat/completions"

client = openai.OpenAI(api_key=API_KEY)

@app.route('/')
def index():
    return send_from_directory('/home/rajs20/mysite', 'index.html')

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(**kwargs):
    return client.completions.create(**kwargs)


@app.route('/process_call_logs', methods=['POST', 'OPTIONS'])
def process_call_logs():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'CORS preflight request success'})
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        response.headers.add('Access-Control-Allow-Origin', '*')  # Allowing any origins
        return response

    # Get the call logs text from the request
    call_logs = request.json.get('call_logs', '')

    # Process the call logs and store the processed document with a timestamp
    processed_document = gpt3_process_call_logs(call_logs)
    timestamp = datetime.now()
    processed_documents[timestamp] = processed_document

    # Return the response
    return jsonify(processed_document)


# Modify the return format of the /get_state_at_timestamp endpoint
@app.route('/get_state_at_timestamp', methods=['GET'])
def get_state_at_timestamp():
    # Get the timestamp from the request parameters
    timestamp_str = request.args.get('timestamp')

    # Convert timestamp string to datetime object
    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M')

    # Find the closest earlier timestamp in processed_documents
    closest_timestamp = min(processed_documents.keys(), key=lambda x: abs(x - timestamp))

    # Retrieve the processed document at the closest timestamp
    processed_document = processed_documents.get(closest_timestamp, {})

    # If there are no processed documents or the timestamp is within the last 24 hours,
    # return an empty response as highlighting is not applicable
    if not processed_document or (datetime.now() - closest_timestamp) < timedelta(days=1):
        return jsonify({'status': 'No data available for the given timestamp'})

    # Get the facts from the processed document at the closest timestamp
    closest_facts = processed_document.get('facts', [])

    # Return the facts along with the timestamp
    return jsonify({'timestamp': closest_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'facts': closest_facts})



@app.route('/add_document', methods=['POST'])
def add_document():
    if request.method == 'POST':
        # Clear previously indexed documents and information
        processed_documents.clear()

        # Get the document URL and question from the request
        document_url = request.json.get('document_url', '')
        question = request.json.get('question', '')

        # Process the document and store it with a timestamp
        processed_document = process_document_from_url(document_url, question)
        timestamp = datetime.now()
        processed_documents[timestamp] = processed_document

        # Return the response
        return jsonify({'message': 'Document added successfully.', 'status': 'processing'})
    else:
        return jsonify({'error': 'Invalid request method.'})


def gpt3_process_call_logs(call_logs):
    # Prompt for GPT-3
    prompt = f"Given the following call logs:\n{call_logs}\nWhat product design decisions did the team make?"

    # Send request to GPT-3 API using backoff decorator
    completion = completions_with_backoff(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0.5,
        max_tokens=500
    )

    # Log the entire completion response
    logging.debug("Completion Response: %s", completion)

    if hasattr(completion, 'choices') and completion.choices:
        # Extract completion text
        completion_text = completion.choices[0].text

        # Split completion text into individual facts
        extracted_facts = completion_text.split('\n')
        logging.debug("Extracted Facts: %s", extracted_facts)
        return {"facts": extracted_facts, "status": "done"}
    else:
        error_message = completion.error or 'Failed to process call logs'
        logging.error("Failed to process call logs: %s", error_message)
        return {"error": error_message}


def process_document_from_url(document_url, question):
    try:
        # Extract timestamp from the document URL
        parsed_url = urlparse(document_url)
        timestamp = extract_timestamp_from_url(parsed_url)

        # Request document content from the provided URL
        response = requests.get(document_url)
        response.raise_for_status()  # Raise exception for non-200 status codes
        document_content = response.text

        # Call function to parse document content and extract facts
        facts = parse_document(document_url, document_content, question)

        return {"timestamp": timestamp, "facts": facts, "status": "done"}
    except Exception as e:
        logging.error("Failed to process document from URL: %s", str(e))
        return {"error": "Failed to process document from URL", "status": "done"}


def parse_document(document_url, document_content, question):
    # Generate facts using the provided question and call logs
    facts = []

    # Construct prompt for GPT-3
    prompt = f"Given the call log:\n{document_content}\n Question: {question}"

    # Send request to GPT-3 API using backoff decorator
    completion = completions_with_backoff(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0.5,
        max_tokens=500
    )

    # Extract completion text
    if hasattr(completion, 'choices') and completion.choices:
        completion_text = completion.choices[0].text
        facts.append(completion_text.strip())

    return facts


def extract_timestamp_from_url(parsed_url):
    # Extract timestamp from the URL path
    timestamp_match = re.search(r'\d{8}_\d{6}', parsed_url.path)
    if timestamp_match:
        return timestamp_match.group()
    else:
        return None


@app.route('/submit_question_and_documents', methods=['POST'])
def submit_question_and_documents():
    if request.method == 'POST':
        # Clear previously indexed documents and information
        processed_documents.clear()

        # Get the question and list of documents from the request
        data = request.json
        question = data.get('question', '')
        documents = data.get('documents', [])

        # Process each document and store the results
        results = []
        for document_url in documents:
            processed_document = process_document_from_url(document_url, question)
            timestamp = datetime.now()
            processed_documents[timestamp] = processed_document
            results.append({'document_url': document_url, 'processed_document': processed_document})

        return jsonify({'message': 'Question and documents submitted successfully.', 'status': 'processing', 'results': results})
    else:
        return jsonify({'error': 'Invalid request method.'})
    
@app.route('/highlight_facts', methods=['GET'])
def highlight_facts():
    # Get the timestamp from the request parameters
    timestamp_str = request.args.get('timestamp')

    # Convert timestamp string to datetime object
    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M')

    # Find the closest earlier timestamp in processed_documents
    closest_timestamp = min(processed_documents.keys(), key=lambda x: abs(x - timestamp))

    # Retrieve the processed document at the closest timestamp
    processed_document = processed_documents.get(closest_timestamp, {})

    # If there are no processed documents or the timestamp is within the last 24 hours,
    # return an empty response as highlighting is not applicable
    if not processed_document or (datetime.now() - closest_timestamp) < timedelta(days=1):
        return jsonify({'added': [], 'removed': []})

    # Get the facts from the processed document at the closest timestamp
    closest_facts = processed_document.get('facts', [])

    # Retrieve the facts from the previous timestamp
    previous_timestamp = max(t for t in processed_documents.keys() if t < closest_timestamp)
    previous_facts = processed_documents.get(previous_timestamp, {}).get('facts', [])

    # Identify added and removed facts
    added_facts = list(set(closest_facts) - set(previous_facts))
    removed_facts = list(set(previous_facts) - set(closest_facts))

    # Return the highlighted facts
    return jsonify({'added': added_facts, 'removed': removed_facts})


if __name__ == '__main__':
    app.run(debug=True)
