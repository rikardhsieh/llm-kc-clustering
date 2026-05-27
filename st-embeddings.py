import os
import json
import csv
from sentence_transformers import SentenceTransformer

INPUT_JSON = 'fragor_2017.json'
OUTPUT_CSV = 'fragor_2017.csv'
MAX_QUESTIONS = None  # sätt till int för begränsning, eller None för alla
MODEL_NAME = 'KBLab/sentence-bert-swedish-cased'


def load_gemini_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fil saknas: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_existing_pairs(csv_path):
    existing = set()
    if not os.path.exists(csv_path):
        return existing

    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get('qid', '').strip()
            skill = row.get('skill', '').strip()
            if qid and skill:
                existing.add((qid, skill))
    return existing


def load_all_existing_rows(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        return rows

    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_csv(csv_path, rows, embedding_dim):
    fieldnames = ['qid', 'skill'] + [f'emb_{i}' for i in range(embedding_dim)]
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    data = load_gemini_data(INPUT_JSON)
    existing_pairs = load_existing_pairs(OUTPUT_CSV)

    model = SentenceTransformer(MODEL_NAME)

    qid_skill_pairs = []
    count_inserted = 0

    for item in data:
        if MAX_QUESTIONS is not None and count_inserted >= MAX_QUESTIONS:
            break

        qid = str(item.get('qid', '')).strip()
        skills = item.get('skills', [])

        if not qid or not isinstance(skills, list):
            continue

        for skill in skills:
            skill_text = str(skill).strip()
            if not skill_text:
                continue

            pair = (qid, skill_text)
            if pair in existing_pairs:
                continue

            qid_skill_pairs.append(pair)
            existing_pairs.add(pair)
            count_inserted += 1

            if MAX_QUESTIONS is not None and count_inserted >= MAX_QUESTIONS:
                break

    if not qid_skill_pairs:
        print('Inget nytt att bearbeta.')
        return

    texts = [t for _, t in qid_skill_pairs]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    new_rows = []
    for (qid, skill_text), emb in zip(qid_skill_pairs, embeddings):
        record = {'qid': qid, 'skill': skill_text}
        for i, val in enumerate(emb):
            record[f'emb_{i}'] = float(val)
        new_rows.append(record)

    # Ladda alla gamla rader och kombinera
    all_rows = load_all_existing_rows(OUTPUT_CSV) + new_rows

    save_csv(OUTPUT_CSV, all_rows, len(embeddings[0]))

    print(f"Sparat {len(new_rows)} rader till {OUTPUT_CSV}.")


if __name__ == '__main__':
    main()
