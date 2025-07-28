print("🚀 Script started...")

import pandas as pd
import json
import faiss
import os
from sentence_transformers import SentenceTransformer

# Setup output directory
os.makedirs("vectorstore", exist_ok=True)

# Clear previous vectorstore
if os.path.exists("vectorstore/faiss_index.index"):
    os.remove("vectorstore/faiss_index.index")
if os.path.exists("vectorstore/metadata.json"):
    os.remove("vectorstore/metadata.json")

# Sheet configurations with row boundaries
sheet_configs = [
    {
        "filename": "tools.xlsx",
        "sheet_name": "tools",
        "embed_cols": [0, 1, 2, 4],
        "display_cols": [0, 1, 2],
        "column_headers": ["Category", "Sub-Category", "Name of Tool"],
        "skip_row": 0,
        "max_rows": 231
    },
    {
        "filename": "service-providers.xlsx",
        "sheet_name": "service-providers",
        "embed_cols": [1],
        "display_cols": [0],
        "column_headers": ["Name of Service Provider"],
        "skip_row": 1,
        "max_rows": 25
    },
    {
        "filename": "training-courses.xlsx",
        "sheet_name": "training-courses",
        "embed_cols": [8, 10, 2, 1, 0],
        "display_cols": [0, 1, 2],
        "column_headers": ["Skill", "Topic", "Course Title"],
        "skip_row": 1,
        "max_rows": 110
    }
]

# Initialize data containers
all_texts = []
all_metadata = []

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Process each sheet
for config in sheet_configs:
    filepath = os.path.join("data", config["filename"])

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        continue

    try:
        # Read Excel with row limit
        df = pd.read_excel(filepath, header=0, nrows=config["max_rows"])
        print(f"📊 Loaded {len(df)} rows from {config['filename']} (max: {config['max_rows']})")
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        continue

    if config["skip_row"] is not None:
        df = df.drop(index=config["skip_row"], errors="ignore").reset_index(drop=True)

    try:
        embed_df = df.iloc[:, config["embed_cols"]]
        display_df = df.iloc[:, config["display_cols"]]
    except IndexError:
        print(f"❌ Column indices out of range in {config['filename']}")
        continue

    # Convert to clean strings and join for embedding
    texts = embed_df.fillna("").astype(str).agg(" ".join, axis=1).tolist()
    display_data = display_df.fillna("").astype(str).values.tolist()

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

# Save metadata
with open("vectorstore/metadata.json", "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=2)

# Save FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
faiss.write_index(index, "vectorstore/faiss_index.index")

print(f"✅ FAISS index built with {len(all_texts)} entries.")
print("📈 Row boundaries applied:")
print("  - Sheet 1: 231 rows")
print("  - Sheet 2: 25 rows") 
print("  - Sheet 3: 110 rows")