import json
import re
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from transformers import pipeline, BertTokenizer, BertModel
from concurrent.futures import ThreadPoolExecutor

nlp = spacy.load("en_core_web_sm")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n', ' ', text)
    return text

def preprocess_documents(documents):
    processed_docs = []
    for doc in documents:
        doc_text = clean_text(doc['content'])
        spacy_doc = nlp(doc_text)
        cleaned_text = " ".join([token.text for token in spacy_doc if not token.is_stop and not token.is_punct])
        processed_docs.append(cleaned_text)
    return processed_docs

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def filter_documents(query, documents, threshold=0.7):
    query_embedding = embed_text(query)
    filtered_docs = []
    for doc in documents:
        doc_embedding = embed_text(doc)
        similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
        if similarity > threshold:
            filtered_docs.append(doc)
    return filtered_docs

def chunk_document(document, chunk_size=512):
    words = document.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def summarize_chunks(chunks):
    summaries = [summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return ' '.join(summaries)

def cross_verify_answers(answers):
    verified_answers = defaultdict(int)
    for answer in answers:
        verified_answers[answer['answer']] += 1
    return max(verified_answers, key=verified_answers.get)

def process_document(document, query):
    chunks = chunk_document(document)
    summary = summarize_chunks(chunks)
    answer = qa_pipeline({'question': query, 'context': summary})
    return answer

def answer_query(query, documents):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_document, doc['content'], query) for doc in documents]
        answers = [future.result() for future in futures]
    verified_answer = cross_verify_answers(answers)
    return verified_answer

def main(query):
    with open('data.json', 'r') as f:
        documents = json.load(f)

    processed_docs = preprocess_documents(documents)
    filtered_docs = filter_documents(query, processed_docs)
    
    if not filtered_docs:
        print("No relevant documents found.")
        return

    answers = answer_query(query, filtered_docs)
    
    print("Final Answer:", answers)

query = "What are the challenges of RAG systems?"
main(query)
