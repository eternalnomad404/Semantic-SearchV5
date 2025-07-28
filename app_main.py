# app_main.py

import os
import faiss
import json
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)

# Load model and precomputed vectorstore
model = SentenceTransformer("all-MiniLM-L6-v2")

VECTORSTORE_DIR = "vectorstore"
INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index.index")
METADATA_PATH = os.path.join(VECTORSTORE_DIR, "metadata.json")

# Ensure files exist
if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
    raise FileNotFoundError("❌ Precomputed FAISS index or metadata not found. Run generate_embeddings.py first.")

# Load index and metadata
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

@app.route("/", methods=["GET", "POST"])
def search():
    results = []
    if request.method == "POST":
        query = request.form["query"]
        query_vector = model.encode([query])
        D, I = index.search(query_vector, 20)

        similarity_threshold = 0.35
        for distance, idx in zip(D[0], I[0]):
            similarity = 1 / (1 + distance)
            if similarity >= similarity_threshold and idx < len(metadata):
                entry = metadata[idx]
                if entry.get("values") and any(str(v).strip() for v in entry["values"]):
                    results.append({
                        "sheet": entry["sheet"],
                        "column_headers": entry["column_headers"],
                        "values": entry["values"],
                        "similarity": round(similarity, 3)
                    })

    return render_template("search_results.html", results=results, query=request.form.get("query", ""))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)  # Production-compatible
