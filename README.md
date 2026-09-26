# RAG LLM Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their content.

## Features

- Upload PDF documents
- Extract text from PDFs
- Split documents into text chunks
- Generate embeddings using Gemini
- Retrieve relevant document chunks using cosine similarity
- Generate answers using Gemini
- Chat interface built with Streamlit
- Clear PDF and chat history
- Uses environment variables for API-key security

## Tech Stack

- Python
- Streamlit
- Google Gemini API
- Gemini Embeddings
- NumPy
- PyPDF
- python-dotenv

## How It Works

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Gemini Embeddings
    ↓
Cosine Similarity Search
    ↓
Relevant Context Retrieval
    ↓
Gemini LLM
    ↓
Generated Answer


Project Structure

llm-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
└── venv/