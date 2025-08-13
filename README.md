# Semantic Search

A semantic search system.

## Prerequisites

To run locally, you need:  
- **Python 3.9+** installed  
- **pip** (Python package manager)  

## Installation & Usage

```bash
# 1️⃣ Clone the repository
git clone https://github.com/eternalnomad404/Semantic-SearchV5.git
cd Semantic-SearchV5

# 2️⃣ Create a virtual environment
python -m venv venv
# Activate it:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 6️⃣ Run the application
streamlit run app_main.py
```

## File Structure

```
├── app_main.py                 # Main Streamlit app
├── generate_embeddings.py      # Generate vector embeddings
├── process_case_studies.py     # Process case studies
├── data/                       # Source data
├── vectorstore/                # Embeddings and indices
├── static/                     # CSS & assets
├── templates/                  # HTML templates
├── .env.example                # Env variables template
└── requirements.txt            # Python dependencies
```
