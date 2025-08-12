# Semantic Search V4 - GROQ Integration

A semantic search system powered by GROQ LLM for intelligent query intent detection.

## Features

- 🤖 **GROQ LLM Integration**: Intelligent query classification using Llama 3.1 model
- 🔍 **Hybrid Search**: Combines semantic embeddings with TF-IDF keyword matching
- 📊 **Multi-Source Data**: Searches across tools, courses, service providers, and case studies
- 🎯 **Smart Intent Detection**: Automatically detects whether users want tools, courses, providers, case studies, or all
- 🚀 **Streamlit UI**: Clean, interactive web interface

## Setup

### 1. Environment Variables

Copy the example environment file and add your API keys:
```bash
cp .env.example .env
```

Edit `.env` and add your GROQ API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get your GROQ API key from: https://console.groq.com/

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Embeddings (First Time Setup)

```bash
python generate_embeddings.py
```

### 4. Run the Application

```bash
streamlit run app_main.py
```

## Deployment

The application is ready for deployment on platforms like Heroku, Railway, or Streamlit Cloud.

### Environment Variables for Production

Set these environment variables in your deployment platform:
- `GROQ_API_KEY`: Your GROQ API key

## File Structure

```
├── app_main.py                 # Main Streamlit application
├── generate_embeddings.py      # Generate vector embeddings
├── process_case_studies.py     # Process case study documents
├── data/                       # Source data files
├── vectorstore/                # Generated embeddings and indices
├── static/                     # CSS and static assets
├── templates/                  # HTML templates
├── .env.example                # Environment variables template
└── requirements.txt            # Python dependencies
```

## Usage

1. **Query Intent Detection**: The system automatically detects if you're looking for:
   - Tools/Software
   - Training Courses
   - Service Providers
   - Case Studies
   - Everything (general search)

2. **Search Results**: Returns relevant results with similarity scores and category filtering

3. **Hybrid Scoring**: Combines semantic similarity (70%) with keyword matching (30%)

## API Integration

The system uses GROQ's Llama 3.1 model for intelligent query classification, providing much more accurate intent detection than traditional keyword matching approaches.
