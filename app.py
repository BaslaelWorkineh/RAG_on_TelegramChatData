import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from colorama import init
from flask_cors import CORS

init(autoreset=True)

app = Flask(__name__)
CORS(app)

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)

def preprocess_data(input_file, output_file, chunk_method='id'):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    messages = data['messages']
    chunked_messages = []

    if chunk_method == 'conversation':
        chunked_messages = chunk_conversations(messages, timedelta(minutes=10))
    elif chunk_method == 'id':
        max_messages_per_chunk = 1
        chunked_messages = chunk_messages_by_id(messages, max_messages_per_chunk)

    output_data = {
        'name': data.get('name', 'Telegram Data'),
        'type': data.get('type', 'data_conversion'),
        'id': data.get('id', 1),
        'messages': chunked_messages
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

def chunk_conversations(messages, max_time_gap):
    conversations = []
    current_conversation = []
    last_message_time = None

    for message in messages:
        message_time = datetime.fromisoformat(message['date'])

        if last_message_time and (message_time - last_message_time > max_time_gap):
            conversations.append(current_conversation)
            current_conversation = []

        current_conversation.append(message)
        last_message_time = message_time

    if current_conversation:
        conversations.append(current_conversation)

    return conversations

def chunk_messages_by_id(messages, max_messages_per_chunk=1):
    chunked_messages = []
    for i in range(0, len(messages), max_messages_per_chunk):
        chunk = [{'from': message['from'], 'text': message['text']} for message in messages[i:i + max_messages_per_chunk]]
        chunked_messages.append(chunk)
    return chunked_messages

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = 'models/embedding-001'
        title = "Telegram Chat History"

        embeddings = []
        for doc in input:
            embedding = genai.embed_content(
                model=model,
                content=[doc],
                task_type="retrieval_document",
                title=title)["embedding"]
            embeddings.append(embedding[0])  # Flatten the nested list
        return embeddings

def create_chroma_db(documents, name):
    chroma_client = chromadb.PersistentClient(path="./database/")

    db = chroma_client.get_or_create_collection(
        name=name, embedding_function=GeminiEmbeddingFunction())

    initial_size = db.count()
    for i, d in enumerate(documents):
        db.add(
            documents=[d],
            ids=[str(i + initial_size)]
        )
    return db

def get_chroma_db(name):
    chroma_client = chromadb.PersistentClient(path="./database/")
    return chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

def get_relevant_passages(query, db, n_results=5):
    passages = db.query(query_texts=[query], n_results=n_results)['documents'][0]
    return passages

def make_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "")
    prompt = f"""question: {query}.\n
    Additional Information:\n {escaped}\n
    If you find that the question is unrelated to the additional information, you can ignore it and respond with 'OUT OF CONTEXT'.\n
    Your response should be a coherent paragraph explaining the answer:\n
    """
    return prompt

def convert_passages_to_paragraph(passages):
    context = ""
    for passage in passages:
        context += passage + "\n"
    return context

def load_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

input_file = 'data.json'
output_file = 'telegram_data.json'
preprocess_data(input_file, output_file, chunk_method='id')

data_file = 'telegram_data.json'
data = load_data_from_json(data_file)

documents = []
for conversation in data['messages']:
    for message in conversation:
        entry = f"{message['from']}: {message['text']}"
        documents.append(entry)

db = create_chroma_db(documents, "sme_db")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    try:
        passages = get_relevant_passages(question, db, n_results=5)
        if passages:
            context = convert_passages_to_paragraph(passages)
            prompt = make_prompt(question, context)
            model = genai.GenerativeModel('gemini-pro')
            answer = model.generate_content(prompt)
            
            return jsonify({
                'question': question,
                'answer': answer.text
            })
        else:
            return jsonify({
                'error': 'No relevant documents found for summarization.'
            })
    except Exception as e:
        return jsonify({
            'error': f'Error occurred: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True, port=8080)
