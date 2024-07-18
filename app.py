import sys
import json
import sqlite3
from flask import Flask, request, jsonify, render_template, redirect, url_for
from telethon.sync import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.functions.messages import GetHistoryRequest
from datetime import datetime
from dotenv import dotenv_values
import asyncio
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from summarizer import preprocess_documents, filter_documents, answer_query


config = dotenv_values(".env")

if 'TELEGRAM_API_ID' not in config or 'TELEGRAM_API_HASH' not in config or 'TELEGRAM_USERNAME' not in config:
    print("Error: Missing TELEGRAM_API_ID, TELEGRAM_API_HASH, or TELEGRAM_USERNAME in .env")
    sys.exit(1)

api_id = int(config['TELEGRAM_API_ID'])
api_hash = config['TELEGRAM_API_HASH']
username = config['TELEGRAM_USERNAME']

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect('sessions.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        phone TEXT PRIMARY KEY,
        session_str TEXT NOT NULL,
        phone_code_hash TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

init_db()

# Custom JSON encoder to handle datetime
class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, bytes):
            return list(o)
        return json.JSONEncoder.default(self, o)

def save_session(phone, session_str, phone_code_hash):
    conn = sqlite3.connect('sessions.db')
    cursor = conn.cursor()
    cursor.execute('REPLACE INTO sessions (phone, session_str, phone_code_hash) VALUES (?, ?, ?)', (phone, session_str, phone_code_hash))
    conn.commit()
    print(f"Session for {phone} saved.")
    conn.close()

def get_session(phone):
    conn = sqlite3.connect('sessions.db')
    cursor = conn.cursor()
    cursor.execute('SELECT session_str, phone_code_hash FROM sessions WHERE phone = ?', (phone,))
    row = cursor.fetchone()
    conn.close()
    return row if row else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_auth', methods=['GET', 'POST'])
def start_auth():
    if request.method == 'POST':
        phone = request.form.get('phone')
        if not phone:
            return jsonify({"error": "Phone number is required"}), 400

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        client = TelegramClient(StringSession(), api_id, api_hash, loop=loop)
        
        try:
            client.connect()
            send_code_request_result = client.send_code_request(phone)
            phone_code_hash = send_code_request_result.phone_code_hash
            session_str = client.session.save()
            print("Session in start: ", session_str)

            save_session(phone, session_str, phone_code_hash)
            print(f"Session for {phone} started. Session string: {session_str}")
            client.disconnect()
            return redirect(url_for('verify_code', phone=phone))
        except Exception as e:
            client.disconnect()
            return jsonify({"error": str(e)}), 500

    return render_template('start_auth.html')

@app.route('/verify_code', methods=['GET', 'POST'])
def verify_code():
    phone = request.args.get('phone')
    if request.method == 'POST':
        phone = request.form.get('phone')
        code = request.form.get('code')
        password = request.form.get('password', None)

        if not phone or not code:
            return jsonify({"error": "Phone and code are required"}), 400

        session_data = get_session(phone)
        if not session_data:
            print(f"Session not found for {phone}")
            return jsonify({"error": "Session not found. Start authentication first."}), 404

        session_str, phone_code_hash = session_data

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        client = TelegramClient(StringSession(session_str), api_id, api_hash, loop=loop)

        try:
            client.connect()
            client.sign_in(phone=phone, code=code, phone_code_hash=phone_code_hash)
        except Exception as e:
            if 'password' in str(e).lower():
                if not password:
                    client.disconnect()
                    return render_template('verify_code.html', phone=phone, error="Password required for two-step verification")
                else:
                    try:
                        client.sign_in(password=password)
                    except Exception as e:
                        client.disconnect()
                        return jsonify({"error": str(e)}), 500
            else:
                client.disconnect()
                return jsonify({"error": str(e)}), 500

        session_str = client.session.save()
        save_session(phone, session_str, phone_code_hash)
        print(f"Session for {phone} verified. Session string: {session_str}")
        client.disconnect()
        return redirect(url_for('dump_messages'))

    return render_template('verify_code.html', phone=phone)


@app.route('/dump_messages', methods=['GET', 'POST'])
def dump_messages():
    if request.method == 'POST':
        phone = request.json.get('phone')
        group_url = request.json.get('group_url')

        if not phone or not group_url:
            return jsonify({"error": "Phone and group URL are required"}), 400

        session_data = get_session(phone)
        if not session_data:
            return jsonify({"error": "Session not found. Authenticate first."}), 404

        session_str, phone_code_hash = session_data

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        client = TelegramClient(StringSession(session_str), api_id, api_hash, loop=loop)

        async def dump_all_messages(channel, short_url_, datetime_):
            offset_msg = 0
            limit_msg = 20
            all_messages = []

            now = datetime.now()
            one_week_ago = now - timedelta(days=7)

            while True:
                history = await client(GetHistoryRequest(
                    peer=channel,
                    offset_id=offset_msg,
                    offset_date=int(one_week_ago.timestamp()),  # Convert datetime to UNIX timestamp
                    add_offset=0,
                    limit=limit_msg,
                    max_id=0,
                    min_id=0,
                    hash=0
                ))
                
                messages = history.messages
                if not messages:
                    break
                
                for message in messages:
                    all_messages.append(message.to_dict())
                
                offset_msg = messages[-1].id
                print(f'{datetime.now()} | Retrieved records: {len(all_messages)}', end='\r')

                # Check conditions to break the loop
                if len(all_messages) >= 1000 or message.date.replace(tzinfo=None) < one_week_ago:
                    break
            

            filename = "data.json"
            with open(filename, 'w', encoding='utf8') as outfile:
                json.dump(all_messages, outfile, ensure_ascii=False, cls=DateTimeEncoder)
            logger.info("Messages dumped to data.json")

        try:
            client.connect()
            channel = client.get_entity(group_url)
            channel_string = group_url.split('/')[-1]
            datetime_string = datetime.now().strftime('%Y%m%dT%H%M%S')
            loop.run_until_complete(dump_all_messages(channel, channel_string, datetime_string))
            client.disconnect()
            logger.info("Disconnected from Telegram")



            # Summarization process
            try:
                query = "What is SingularityNET<?"  # Example query for summarization
                with open('data.json', 'r') as f:
                    documents = json.load(f)
                logger.info("Loaded data.json")

                processed_docs = preprocess_documents(documents)
                logger.info("Documents preprocessed")
                filtered_docs = filter_documents(query, processed_docs)
                logger.info("Documents filtered")

                if not filtered_docs:
                    return jsonify({"error": "No relevant documents found for summarization."}), 404

                answers = answer_query(query, filtered_docs)
                logger.info("Query answered")

                return jsonify({"summary": answers['answer']}), 200

            except Exception as e:
                logger.error(f"Error in summarization: {str(e)}")
                return jsonify({"error": f"Error in summarization: {str(e)}"}), 500


        
        except Exception as e:
            client.disconnect()
            logger.error(f"Error: {str(e)}")

            return jsonify({"error": str(e)}), 500

    return render_template('dump_messages.html')


if __name__ == '__main__':
    app.run(debug=True)
