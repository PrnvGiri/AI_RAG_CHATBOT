# 🤖 Simple AI RAG Chatbot

A minimal RAG (Retrieval-Augmented Generation) application that reads `RayOptics.pdf` and launches an interactive Gradio Chat UI using **Google Gemini** and **LangChain**.

---

## ⚡ Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Add API Key to `.env`
Create a `.env` file in this directory and add your key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Run the Chatbot
```bash
python3 rag_chatbot.py
```

That's it! It automatically loads `RayOptics.pdf`, builds the embeddings, and launches the Gradio Web Chat Interface in your browser.

---

## 📁 Repository Files

- **[rag_chatbot.py](file:///Users/pranav/PRNV/Programs/AI_RAG_ChatBot/rag_chatbot.py)**: Clean & simple Python script.
- **[RayOptics.pdf](file:///Users/pranav/PRNV/Programs/AI_RAG_ChatBot/RayOptics.pdf)**: Target PDF document.
- **[requirements.txt](file:///Users/pranav/PRNV/Programs/AI_RAG_ChatBot/requirements.txt)**: Python dependencies.
- **[RagImplementation7thJune2026.ipynb](file:///Users/pranav/PRNV/Programs/AI_RAG_ChatBot/RagImplementation7thJune2026.ipynb)**: Original Jupyter Notebook.
