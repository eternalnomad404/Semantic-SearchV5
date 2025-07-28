import os
import shutil
import faiss
import json
import pandas as pd
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)

# === Auto generate embeddings ===

model = SentenceTransformer("all-MiniLM-L6-v2")

data_folder = "data"
vectorstore_folder = "vectorstore"

# Clean up old vectorstore
if os.path.exists(vectorstore_folder):
    shutil.rmtree(vectorstore_folder)
os.makedirs(vectorstore_folder)

index = faiss.IndexFlatL2(384)  # 384-dim for MiniLM
metadata = []

# Loop through all Excel files
for file in os.listdir(data_folder):
    if file.endswith(".xlsx"):
        filepath = os.path.join(data_folder, file)
        xls = pd.ExcelFile(filepath)
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name).fillna("")
            for i, row in df.iterrows():
                row_text = " ".join(str(cell) for cell in row.values)
                embedding = model.encode([row_text])[0]
                index.add(embedding.reshape(1, -1))
                metadata.append({
                    "sheet": f"{file} - {sheet_name}",
                    "column_headers": list(df.columns),
                    "values": list(row.values)
                })

# Save index and metadata
faiss.write_index(index, os.path.join(vectorstore_folder, "faiss_index.index"))
with open(os.path.join(vectorstore_folder, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

# === End embedding generation ===

# Load index and metadata
index = faiss.read_index(os.path.join(vectorstore_folder, "faiss_index.index"))
with open(os.path.join(vectorstore_folder, "metadata.json"), "r", encoding="utf-8") as f:
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
    app.run(debug=True)
