# AI RAG Chatbot (Ray Optics PDF Question Answering System)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system built with Python, LangChain, Google Gemini API, Chroma vector database, and Gradio. The system processes a PDF document ("RayOptics.pdf"), embeds text chunks into a vector database, and answers user questions based strictly on the retrieved context via a Gradio web interface.

---

## What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is an architectural pattern in Artificial Intelligence that enhances Large Language Models (LLMs) by retrieving relevant information from an external document store before generating a response.

Standard LLMs can suffer from two main issues:
1. Lack of domain-specific context or private document knowledge.
2. Hallucination (generating plausible but incorrect answers).

RAG resolves these issues through a three-stage process:
1. **Retrieval**: Given a user query, the system searches a vector store for text segments from the document that are semantically similar to the query.
2. **Augmentation**: The system inserts the retrieved document segments into the system prompt alongside the user query.
3. **Generation**: The LLM reads the context-augmented prompt and generates an accurate response based strictly on the provided document facts.

---

## System Architecture and Data Flow

The pipeline implemented in this project follows this architecture:

```
+------------------+
|  RayOptics.pdf   |
+------------------+
         |
         v
+------------------+
|   PyPDFLoader    |  (Extract text from PDF pages)
+------------------+
         |
         v
+------------------------------------+
| RecursiveCharacterTextSplitter     |  (Chunk size: 2000, Overlap: 500)
+------------------------------------+
         |
         v
+------------------------------------+
| GoogleGenerativeAIEmbeddings       |  (Model: gemini-embedding-001)
+------------------------------------+
         |
         v
+------------------------------------+
|  Chroma Vector Database (In-Mem)   |  (Store document vectors)
+------------------------------------+
         |
         v
+------------------------------------+
|  Vector Search Retriever           |  (Find top matching text chunks)
+------------------------------------+
         |
         v
+------------------------------------+
|  ChatPromptTemplate & Context      |  (Combine retrieved chunks + query)
+------------------------------------+
         |
         v
+------------------------------------+
|  ChatGoogleGenerativeAI            |  (Model: gemini-2.5-flash)
+------------------------------------+
         |
         v
+------------------------------------+
|  Gradio Web Interface              |  (Interactive Web Chat UI)
+------------------------------------+
```

---

## Step-by-Step Guide to Run

Follow these step-by-step instructions to set up and run the application on your machine.

### Step 1: Clone the Repository
Open your terminal and clone the repository:
```bash
git clone https://github.com/PrnvGiri/AI_RAG_CHATBOT.git
cd AI_RAG_CHATBOT
```

### Step 2: Install Required Dependencies
Install all required Python packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

Required packages:
- `langchain`
- `langchain-community`
- `langchain-google-genai`
- `chromadb`
- `pypdf`
- `gradio`
- `python-dotenv`

### Step 3: Set Up Environment Variables
Create a file named `.env` in the root directory of the project. Add your Google Gemini API key:
```env
GOOGLE_API_KEY=your_actual_google_gemini_api_key_here
```
Note: You can get a free Gemini API Key from Google AI Studio.

### Step 4: Verify the Target PDF
Ensure `RayOptics.pdf` is located in the root directory of the project.

### Step 5: Execute the Python Script
Run the Python application:
```bash
python3 rag_chatbot.py
```

The script will automatically:
1. Read `GOOGLE_API_KEY` from your `.env` file.
2. Load and split `RayOptics.pdf` into text chunks.
3. Embed chunks into the Chroma vector database.
4. Launch the Gradio Web Chat Interface.

### Step 6: Access the Web Chatbot
Open the URL printed in the terminal (such as `http://127.0.0.1:7860` or the public Gradio link) in your browser to ask questions about Ray Optics.

---

## Project Structure

```
AI_RAG_CHATBOT/
├── .env                              # Local environment file containing API keys (git-ignored)
├── .gitignore                        # Git ignore patterns
├── README.md                         # Project documentation
├── RagImplementation7thJune2026.ipynb # Original Jupyter Notebook reference
├── RayOptics.pdf                     # Target PDF document
├── rag_chatbot.py                    # Main RAG application script
└── requirements.txt                  # Python dependencies manifest
```

---

## Technical Details

- **Embedding Model**: `gemini-embedding-001` (Task type: `RETRIEVAL_DOCUMENT`)
- **LLM Model**: `gemini-2.5-flash`
- **Vector DB**: Chroma (In-memory)
- **Text Splitter**: Recursive Character Splitter (Chunk size: 2000 characters, Overlap: 500 characters)
- **UI Framework**: Gradio ChatInterface
