# %%
from google import genai
import time
import psycopg2
import os
import json
from dotenv import load_dotenv
from src.llm import extract_skills_with_gemini

load_dotenv()  # laddar miljövariabler från .env-filen, inklusive GEMINI_API_KEY

# %%
conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/test_items")
cursor = conn.cursor()
# SQL-fråga med JOIN för att koppla frågor till provets datum
query = """
    SELECT q.*
    FROM sat_questions q
    JOIN sat_tests t ON q.test_id = t.id
    WHERE t.year = %s;
"""
cursor.execute(query, (2017,))


colnames = [desc[0] for desc in cursor.description]
questions = cursor.fetchall()
conn.close()

# %%
# Ange hur många frågor som ska bearbetas
num_questions = 280

# Ange outputfil
output_file = 'fragor_2017.json'

# Ladda tidigare sparade qids (om fil finns) för att undvika dubbelbearbetning
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        existing_results = json.load(f)
    # Skippa endast qids som redan är färdiga (status==done)
    existing_qids = {
        item['qid'] for item in existing_results
        if item.get('status') == 'done' and 'qid' in item
    }
else:
    existing_results = []
    existing_qids = set()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Använd dictionary för att hålla unika qid och uppdatera befintlig post
results_by_qid = {item['qid']: item for item in existing_results if 'qid' in item}

# %%
count = min(num_questions, len(questions))
for i in range(count):
    question = dict(zip(colnames, questions[i]))
    qid = question.get('id', f'q{i}')

    if qid in existing_qids:
        continue

    start_time = time.time()
    try:
        skills = extract_skills_with_gemini(question, client)
        if skills:
            status = 'done'
        else:
            status = 'retry'

        result = {
            'qid': qid,
            'skills': skills,
            'status': status,
        }

        results_by_qid[qid] = result
        if status == 'done':
            existing_qids.add(qid)

    except Exception as e:
        # Spara felstatus men fortsätt
        result = {
            'qid': qid,
            'skills': [],
            'status': 'retry',
            'error': str(e)
        }
        results_by_qid[qid] = result

# Konvertera till lista innan sparning
all_results = list(results_by_qid.values())
# %%
# Skriv ut till fil (ingen terminaloutput önskas efter körning)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
