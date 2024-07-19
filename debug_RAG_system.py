import os
from dotenv import load_dotenv
import json
import logging
from pprint import pprint

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

import google.generativeai as genai
from colorama import init, Fore, Style

init(autoreset=True)

logging.basicConfig(level=logging.DEBUG)

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)

def preprocess_data(input_file, output_file):
    logging.debug(Fore.YELLOW + f"Starting to preprocess data from {input_file} to {output_file}")
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
    logging.debug(Fore.YELLOW + f"Preprocessing complete. Data written to {output_file}")

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
    logging.debug(Fore.YELLOW + f"Creating Chroma DB with the name {name}")
    chroma_client = chromadb.PersistentClient(path="./database/")

    db = chroma_client.get_or_create_collection(
        name=name, embedding_function=GeminiEmbeddingFunction())

    initial_size = db.count()
    logging.debug(Fore.YELLOW + f"Initial size of the DB: {initial_size}")
    for i, d in enumerate(documents):
        db.add(
            documents=[d],
            ids=[str(i + initial_size)]
        )
    logging.debug(Fore.YELLOW + f"Documents added to the DB. New size: {db.count()}")
    return db

def get_chroma_db(name):
    logging.debug(Fore.YELLOW + f"Getting Chroma DB with the name {name}")
    chroma_client = chromadb.PersistentClient(path="./database/")
    return chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

def get_relevant_passages(query, db, n_results=5):
    logging.debug(Fore.YELLOW + f"Querying the DB for relevant passages with the query: {query}")
    passages = db.query(query_texts=[query], n_results=n_results)['documents'][0]
    logging.debug(Fore.YELLOW + f"Relevant passages found: {passages}")
    return passages

def make_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "")
    prompt = f"""question: {query}.\n
    Additional Information:\n {escaped}\n
    If you find that the question is unrelated to the additional information, you can ignore it and respond with 'Sorry, I couldn't answer that since my data is limited'.\n
    Your response should be a coherent paragraph explaining the answer:\n
    """
    logging.debug(Fore.YELLOW + f"Generated prompt: {prompt}")
    return prompt

def convert_passages_to_paragraph(passages):
    context = ""
    for passage in passages:
        context += passage + "\n"
    logging.debug(Fore.YELLOW + f"Converted passages to paragraph: {context}")
    return context

def load_data_from_json(file_path):
    logging.debug(Fore.YELLOW + f"Loading data from JSON file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    logging.debug(Fore.YELLOW + f"Data loaded: {data}")
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
logging.debug(Fore.YELLOW + f"Documents prepared for DB: {documents}")

db = create_chroma_db(documents, "sme_db")

while True:
    print(Fore.GREEN + "Good Question: What is SingularityNET and what is it's mission?")
    print(Fore.GREEN + "Good Question: What is SingularityNet's ecosystem doing?")
    print(Fore.GREEN + "Good Question: Will AGIX reach 1 dollar?")
    print(Fore.RED + "Bad Question: What happened to Donald Trump?\n")
    question = input("Ask question related to SingularityNET: ")

    try:
        passages = get_relevant_passages(question, db, n_results=5)
        if passages:
            context = convert_passages_to_paragraph(passages)
            prompt = make_prompt(question, context)
            model = genai.GenerativeModel('gemini-pro')
            answer = model.generate_content(prompt)
            
            logging.info(Fore.CYAN + "Retrieved texts:")
            for passage in passages:
                logging.info(Fore.CYAN + passage)
            print('\n\n')
            print(Fore.GREEN + answer.text + "\n")
        else:
            logging.error(Fore.RED + 'No relevant documents found for summarization.')
            print({'error': 'No relevant documents found for summarization.'})
    except Exception as e:
        logging.error(Fore.RED + f"Error occurred: {str(e)}")
        print({'error': f'Error occurred: {str(e)}'})