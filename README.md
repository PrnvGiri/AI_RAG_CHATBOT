# 📄 Simple RAG Chatbot (PDF Question Answering)

A simple Python application that reads any PDF file and lets you ask questions about it using **Google Gemini** and **LangChain**.

---

## 🚀 Quick Setup

### 1. Install Dependencies

Run this command in your terminal:

```bash
pip install -r requirements.txt
```

### 2. Set Your Google API Key

**Linux / macOS:**
```bash
export GOOGLE_API_KEY="your_gemini_api_key_here"
```

**Windows:**
```cmd
set GOOGLE_API_KEY="your_gemini_api_key_here"
```

---

## 💡 How to Run

Run the python file:

```bash
python rag_chatbot.py
```

1. Enter the path to your PDF file when prompted.
2. Choose whether to run in **Terminal (CLI)** or **Web Interface (Gradio UI)**.

---

## 📁 Files in This Project

- **[rag_chatbot.py](file:///Users/pranav/PRNV/Programs/AI_RAG_ChatBot/rag_chatbot.py)**: Python script to load PDF, build RAG chain, and chat.
- **[requirements.txt](file:///Users/pranav/PRNV/Programs/AI_RAG_ChatBot/requirements.txt)**: List of required Python packages.
- **[RagImplementation7thJune2026.ipynb](file:///Users/pranav/PRNV/Programs/AI_RAG_ChatBot/RagImplementation7thJune2026.ipynb)**: Original Jupyter Notebook.
