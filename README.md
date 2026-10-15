# Folio — Chat with your PDFs

Folio is a local PDF question-answering app. Upload a PDF, review its extracted
text, generate a summary, create searchable embeddings, and ask questions with
answers grounded in the document.

## Features

- Extracts selectable text from uploaded PDFs and splits it into passages.
- Generates a summary with Ollama's `llama3.2` model.
- Stores semantic embeddings locally in ChromaDB.
- Answers questions using relevant passages and shows their source pages.
- Includes suggested questions and download options for the PDF and summary.
- Uses a responsive, warm-paper interface designed for comfortable reading.

## Requirements

- Python 3.10 or newer.
- [Ollama](https://ollama.com/) installed and running locally.
- The `llama3.2` model available in Ollama.

Install and download the model:

```powershell
ollama pull llama3.2
```

## Setup

From the project directory, create and activate a virtual environment, then
install the Python dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run the commands below
without activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

## Use

1. Upload a PDF with selectable text.
2. Optionally preview the extracted passages or create a summary.
3. Select **Create Embeddings** to make the document searchable.
4. Ask a question or choose one of the suggested questions.

Scanned PDFs without a text layer are not OCR-processed; they need OCR before
their contents can be searched.

## Local data and privacy

ChromaDB stores embeddings in the local `chroma_db/` directory. The database,
virtual environment, local secrets, and PDF files are excluded from Git. PDF
text and questions are sent to the locally running Ollama service for
summarization and answers.
