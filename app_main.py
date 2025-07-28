from flask import Flask, request, render_template
import faiss
import json
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Make zip function available in Jinja2 templates
app.jinja_env.globals.update(zip=zip)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and metadata
index = faiss.read_index("vectorstore/faiss_index.index")
with open("vectorstore/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

@app.route("/", methods=["GET", "POST"])
def search():
    results = []
    if request.method == "POST":
        query = request.form["query"]
        query_vector = model.encode([query])
        D, I = index.search(query_vector, 20)  # Still search for 20 to get candidates
        
        print(f"Query: {query}")
        print(f"Distances: {D[0][:5]}")  # First 5 distances
        print(f"Indices: {I[0][:5]}")   # First 5 indices
        
        # Set similarity threshold - keeping it reasonable for your data
        similarity_threshold = 0.35  # Based on your debug output, this should work well
        
        for distance, idx in zip(D[0], I[0]):
            # Convert distance to similarity (FAISS returns L2 distance)
            similarity = 1 / (1 + distance)  # Convert to similarity score
            
            print(f"Distance: {distance}, Similarity: {similarity:.3f}, Above threshold: {similarity >= similarity_threshold}")
            
            # Only include results above threshold
            if similarity >= similarity_threshold and idx < len(metadata):
                entry = metadata[idx]
                
                # Check if the entry has meaningful data (not empty)
                has_meaningful_data = False
                if entry.get("values") and len(entry["values"]) > 0:
                    # Check if any of the values are not empty/None
                    for value in entry["values"]:
                        if value and str(value).strip():  # Not empty, None, or just whitespace
                            has_meaningful_data = True
                            break
                
                if has_meaningful_data:
                    print(f"Adding result from sheet: {entry['sheet']} with values: {entry['values'][:3]}")
                    results.append({
                        "sheet": entry["sheet"],
                        "column_headers": entry["column_headers"],
                        "values": entry["values"],
                        "similarity": round(similarity, 3)
                    })
                else:
                    print(f"Skipping empty result from sheet: {entry['sheet']}")
        
        print(f"Final results count: {len(results)}")
    
    return render_template("search_results.html", results=results, query=request.form.get("query", ""))

if __name__ == "__main__":
    app.run(debug=True)