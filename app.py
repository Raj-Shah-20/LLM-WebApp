import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import backoff
import logging
import re
from urllib.parse import urlparse
import os

# Define a dictionary to store processed documents with timestamps
processed_documents = {}

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("OPENAI_API_KEY")

# GPT-3.5 API key and endpoint
GPT_API = "https://api.openai.com/v1/chat/completions"

@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3, max_time=60)
def process_call_logs(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "model": "gpt-3.5-turbo-instruct",  # Using the instruct model
        "max_tokens": 500,
        "temperature": 0.2
    }
    response = requests.post(GPT_API, json=data, headers=headers)
    response.raise_for_status()
    return response.json()

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

@app.route('/get_question_and_facts', methods=['GET'])
def get_question_and_facts():
    # Check if processing is complete
    if processed_documents:
        # Retrieve the latest processed document
        latest_timestamp = max(processed_documents.keys())
        latest_document = processed_documents[latest_timestamp]

        # Check if processing is still ongoing
        if latest_document.get('status') == 'processing':
            return jsonify({'question': latest_document['question'], 'facts': None, 'status': 'processing'})

        # Processing is complete, return the question and facts
        return jsonify({'question': latest_document['question'], 'facts': latest_document['facts'], 'status': 'done'})
    else:
        return jsonify({'error': 'No processed documents available.', 'status': 'error'})

def process_document_from_url(document_url, question):
    try:

        # Request document content from the provided URL
        response = requests.get(document_url)
        response.raise_for_status()  # Raise exception for non-200 status codes
        document_content = response.text

        # Call function to parse document content and extract facts
        facts = parse_document(document_content, question)

        return {"question": question, "facts": facts, "status": "done"}
    except Exception as e:
        logging.error("Failed to process document from URL: %s", str(e))
        return {"error": "Failed to process document from URL", "status": "done"}

def parse_document(document_content, question):
    # Generate facts using the provided question and call logs
    prompt = f"Question: {question}\nDocument: {document_content}"
    facts = process_call_logs(prompt)

    if 'choices' in facts and facts['choices']:
        return [choice['text'].strip() for choice in facts['choices']]
    else:
        return []

if __name__ == '__main__':
    app.run(debug=True)