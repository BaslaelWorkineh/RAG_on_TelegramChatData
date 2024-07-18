import os
from dotenv import load_dotenv
import json

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

def preprocess_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    messages = []
    for message in data['messages']:
        formatted_message = {
            'id': message.get('id', ''),
            'type': message.get('type', ''),
            'date': message.get('date', ''),
            'from': message.get('from', ''),
            'text': message.get('text', '')
        }
        messages.append(formatted_message)
    
    output_data = {
        'name': data.get('name', 'Telegram Data'),
        'type': data.get('type', 'data_conversion'),
        'id': data.get('id', 1),
        'messages': messages
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

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
preprocess_data(input_file, output_file)

data_file = 'telegram_data.json' 
data = load_data_from_json(data_file)

documents = []
for message in data['messages']:
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
