# 📚 StudyMate — Chat with your PDFs

StudyMate is a Streamlit-powered app that lets you **upload PDFs or text files** and then **ask questions** about them.  
It uses **FAISS** for semantic search, **transformers embeddings** for document indexing, and can optionally use the **Gemini API** for more natural answers.

---

## 🚀 Features
- Upload multiple **PDF** or **TXT** files.
- Extract and chunk text automatically.
- Build a FAISS index with **transformers embeddings**.
- Retrieve top relevant passages for a query.
- Answer synthesis in **offline mode** (extractive).
- Optional **Gemini API** integration for natural LLM answers.
- Download citations for your session.

---

## 📦 Installation & Run (All-in-One)

# 1. Clone the repository
```bash
git clone https://github.com/<your-username>/studymate.git
cd studymate
```

# 2. Create a virtual environment
```bash
python -m venv venv
```

# 3. Activate it
# Windows (PowerShell)
```bash
venv\Scripts\activate
```
# macOS/Linux
```bash
source venv/bin/activate
```

# 4. Install dependencies
```bash
pip install streamlit PyPDF2 transformers torch faiss-cpu numpy requests
```

# 5. (Optional) Set Gemini API Key
# Windows (PowerShell)
```bash
$env:GEMINI_API_KEY="your_api_key_here"
```

# macOS/Linux
```bash
export GEMINI_API_KEY="your_api_key_here"
```

# 6. Run the app

📂 Project Structure
```bash
streamlit run app.py

  ├── app.py               # Main Streamlit application
  ├── requirements.txt     # Python dependencies
  ├── README.md            # Project documentation

```

🖼️ Screenshots
---
<img width="1918" height="872" alt="image" src="https://github.com/user-attachments/assets/eaa4edc2-cc23-41ed-88c5-3227657e17f5" />
<img width="1918" height="866" alt="image" src="https://github.com/user-attachments/assets/8ee58c31-5281-4e53-8433-7ae63721eb32" />

⚠️ Notes

The first time you build the index, downloading and loading the model may take 1–2 minutes.
The embedding model can be changed by setting the EMBED_MODEL environment variable.
If FAISS fails on GPU, use faiss-cpu.
---
