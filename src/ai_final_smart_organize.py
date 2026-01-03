#!/usr/bin/env python3
import os
import shutil
import re
import json
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
from openai import OpenAI

USER = "satyamagarwal"

TARGET_DIRS = [
    f"/Users/{USER}/Downloads",
    f"/Users/{USER}/Documents"
]

DEST_NAME = "Organized_AI"
LOG_FILE = f"/Users/{USER}/ai_organize_log.txt"
PROGRESS_FILE = f"/Users/{USER}/ai_progress.json"

AI_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"

BATCH_LIMIT = 1200           
CLUSTER_THRESHOLD = 0.78     
BUDGET_LIMIT = 0.80          

client = OpenAI()


def save_progress(data):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f)


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"done": []}


def local_classify(filename, path):
    name = filename.lower()
    path_l = path.lower()

    if "organized_ai" in path_l:
        return None

    if "whatsapp" in path_l: return "Photos/WhatsApp"
    if "telegram" in path_l: return "Photos/Telegram"
    if "iphone" in path_l: return "Photos/iPhone"
    if "screenshot" in name or "screen shot" in name: return "Screenshots"

    keywords = {
        "invoice": "Documents/Invoices",
        "bill": "Documents/Bills",
        "tax": "Documents/Tax",
        "insurance": "Documents/Insurance",
        "medical": "Documents/Medical",
        "prescription": "Documents/Medical",
        "bank": "Documents/Bank",
        "statement": "Documents/Bank",
        "passport": "Documents/Identity",
        "visa": "Documents/Identity",
        "ticket": "Travel/Tickets",
        "travel": "Travel/General",
        "resume": "Career/Resume",
        "cv": "Career/Resume",
        "offer": "Career/Offers",
        "agreement": "Legal/Agreements",
        "contract": "Legal/Agreements",
        "certificate": "Certificates"
    }

    for k,v in keywords.items():
        if k in name:
            return v

    return None


def get_embedding(text):
    r = client.embeddings.create(
        model=EMBED_MODEL,
        input=text[:2000]
    )
    return np.array(r.data[0].embedding)


def cosine(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))


def ask_ai_for_cluster_name(files):
    names = [Path(f).name for f in files]

    prompt = f"""
You are a macOS file organizer.
These files belong together. Give ONE best folder path like:

Work/Project ABC
Kids/Health
Medical/Reports
Travel/Paris Trip
Banking
Legal
Personal Docs

Only return the folder path, nothing else.
Files:
{json.dumps(names)}
"""

    r = client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {"role":"system","content":"Classify files logically."},
            {"role":"user","content":prompt}
        ]
    )

    return r.choices[0].message.content.strip()


def main():
    progress = load_progress()
    already_done = set(progress["done"])

    files = []

    print("\nScanning files...\n")
    for ROOT in TARGET_DIRS:
        dest = os.path.join(ROOT, DEST_NAME)
        os.makedirs(dest, exist_ok=True)

        for dirpath,_,fs in os.walk(ROOT):
            if DEST_NAME in dirpath:
                continue

            for f in fs:
                full = os.path.join(dirpath, f)
                if full in already_done:
                    continue
                files.append(full)

    if len(files)==0:
        print("Nothing to do.")
        return

    files = files[:BATCH_LIMIT]
    print(f"Processing {len(files)} files this run.")

    remaining_for_ai = []

    print("\nPhase 1: Local smart sort\n")
    for f in tqdm(files):
        name = Path(f).name
        parent = str(Path(f).parent)
        guess = local_classify(name,parent)

        if guess:
            root = Path(f).parents[1]
            dest = os.path.join(str(root), DEST_NAME, guess)
            os.makedirs(dest, exist_ok=True)
            shutil.move(f, os.path.join(dest,name))
            progress["done"].append(f)
        else:
            remaining_for_ai.append(f)

    print(f"\nRemaining requiring AI: {len(remaining_for_ai)}")

    if len(remaining_for_ai)==0:
        save_progress(progress)
        print("Done.")
        return

    print("\nPhase 2: Cheap AI grouping\n")

    items=[]
    for f in tqdm(remaining_for_ai):
        text = Path(f).name
        emb = get_embedding(text)
        items.append({"file":f,"emb":emb})

    clusters=[]
    for item in items:
        placed=False
        for c in clusters:
            if cosine(item["emb"], c[0]["emb"]) > CLUSTER_THRESHOLD:
                c.append(item)
                placed=True
                break
        if not placed:
            clusters.append([item])

    print(f"AI clusters to label: {len(clusters)}")

    cost_per_cluster = 0.001
    max_clusters_allowed = int(BUDGET_LIMIT / cost_per_cluster)

    if len(clusters) > max_clusters_allowed:
        print(f"Too many clusters. Budget allows labeling only {max_clusters_allowed}.")
        clusters = clusters[:max_clusters_allowed]
    else:
        print("Within budget. Proceeding.")

    for cluster in clusters:
        label = ask_ai_for_cluster_name([i["file"] for i in cluster])

        for item in cluster:
            f=item["file"]
            name=Path(f).name
            root=Path(f).parents[1]
            dest=os.path.join(str(root), DEST_NAME, label)
            os.makedirs(dest, exist_ok=True)
            shutil.move(f, os.path.join(dest,name))
            progress["done"].append(f)

    save_progress(progress)
    print("\nDONE under budget.\n")


if __name__=="__main__":
    main()
