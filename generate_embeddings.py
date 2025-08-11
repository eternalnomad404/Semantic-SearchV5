print("🚀 Script started...")

import pandas as pd
import json
import faiss
import os
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
import numpy as np
import pickle

# Setup output directory
os.makedirs("vectorstore", exist_ok=True)

# Clear previous vectorstore
if os.path.exists("vectorstore/faiss_index.index"):
    os.remove("vectorstore/faiss_index.index")
if os.path.exists("vectorstore/metadata.json"):
    os.remove("vectorstore/metadata.json")
if os.path.exists("vectorstore/tfidf.pkl"):
    os.remove("vectorstore/tfidf.pkl")

# Sheet configurations with row boundaries
sheet_configs = [
    {
        "filename": "tools.xlsx",
        "sheet_name": "Cleaned Sheet",  # Changed from "tools" to actual sheet name
        "embed_cols": [0, 1, 2, 4],
        "display_cols": [0, 1, 2],
        "column_headers": ["Category", "Sub-Category", "Name of Tool"],
        "skip_rows": 0,  # No need to skip rows in cleaned sheet
        "max_rows": 231
    },
    {
        "filename": "service-providers.xlsx",
        "sheet_name": "Service Provider Profiles",
        "embed_cols": [0, 1],
        "display_cols": [0],
        "column_headers": ["Name of Service Provider"],
        "skip_rows": 0,
        "max_rows": 25
    },
    {
        "filename": "training-courses.xlsx",
        "sheet_name": "Training Program",
        "embed_cols": [8, 10, 2, 1, 0],
        "display_cols": [0, 1, 2],
        "column_headers": ["Skill", "Topic", "Course Title"],
        "skip_rows": 0,
        "max_rows": 110
    }
]

# Initialize data containers
all_texts = []
all_metadata = []
raw_texts = []  # For TF-IDF

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Process each sheet
for config in sheet_configs:
    filepath = os.path.join("data", config["filename"])

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        continue

    try:
        # Read Excel with proper sheet and skip_rows handling
        if "skip_rows" in config and config["skip_rows"] > 0:
            # Skip metadata rows if needed
            df = pd.read_excel(filepath, sheet_name=config["sheet_name"], header=0, skiprows=config["skip_rows"], nrows=config["max_rows"])
        else:
            # Normal reading (for cleaned sheets with proper headers)
            df = pd.read_excel(filepath, sheet_name=config["sheet_name"], header=0, nrows=config["max_rows"])
        print(f"📊 Loaded {len(df)} rows from {config['filename']} sheet '{config['sheet_name']}' (max: {config['max_rows']})")
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        continue

    # Remove the old skip_row logic since we now use skiprows in read_excel

    try:
        embed_df = df.iloc[:, config["embed_cols"]]
        display_df = df.iloc[:, config["display_cols"]]
    except IndexError:
        print(f"❌ Column indices out of range in {config['filename']}")
        continue

    # Convert to clean strings and join for embedding
    texts = embed_df.fillna("").astype(str).agg(" ".join, axis=1).tolist()
    display_data = display_df.fillna("").astype(str).values.tolist()
    
    # Store raw texts for TF-IDF
    raw_texts.extend(texts)

    for row_data in display_data:
        metadata_entry = {
            "sheet": config["sheet_name"],
            "column_headers": config["column_headers"],
            "values": row_data
        }
        all_metadata.append(metadata_entry)

    all_texts.extend(texts)

# Generate and save embeddings
print(f"🧠 Generating embeddings for {len(all_texts)} rows...")
embeddings = model.encode(all_texts)

# Generate TF-IDF vectors
print("📊 Generating TF-IDF vectors...")
tfidf = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    ngram_range=(1, 2)
)
tfidf_vectors = tfidf.fit_transform(raw_texts)

# Save TF-IDF vectorizer and vectors
with open("vectorstore/tfidf.pkl", "wb") as f:
    pickle.dump({
        'vectorizer': tfidf,
        'vectors': tfidf_vectors
    }, f)

# Save metadata
with open("vectorstore/metadata.json", "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=2)

# Save FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
faiss.write_index(index, "vectorstore/faiss_index.index")

print(f"✅ FAISS index built with {len(all_texts)} entries.")
print(f"✅ TF-IDF vectors generated with {tfidf_vectors.shape[1]} features.")
print("📈 Row boundaries applied:")
print("  - Sheet 1: 231 rows")
print("  - Sheet 2: 25 rows") 
print("  - Sheet 3: 110 rows")