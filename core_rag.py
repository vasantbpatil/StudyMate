import fitz  # PyMuPDF
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- Model Initialization ---

# Embedding model for converting text to vectors
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Lazy-load the Language Model to save resources
_llm = None
_tokenizer = None

def _load_llm():
    """Lazily loads the language model and tokenizer when first needed."""
    global _llm, _tokenizer
    if _llm is None or _tokenizer is None:
        model_name = "google/flan-t5-base"
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return _tokenizer, _llm

def parse_pdf_to_contents(pdf_path, min_length=50):
    """
    Parses a PDF and extracts its content into clean, readable paragraphs.
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"
        
        # Split text into paragraphs based on one or more newline characters
        paragraphs = re.split(r'\n\s*\n', full_text)
        
        # Clean up paragraphs and filter out short, irrelevant ones
        cleaned_contents = []
        for para in paragraphs:
            # Normalize whitespace and strip leading/trailing spaces
            cleaned_para = re.sub(r'\s+', ' ', para).strip()
            if len(cleaned_para) >= min_length:
                cleaned_contents.append(cleaned_para)
        
        return cleaned_contents if cleaned_contents else [full_text] # Fallback to full text
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return []

def create_faiss_index(contents):
    """
    Creates a FAISS index from the provided content for efficient searching.
    Uses cosine similarity.
    """
    if not contents:
        return None
    try:
        # Generate embeddings and normalize them for cosine similarity search
        embeddings = embedding_model.encode(contents, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Create a FAISS index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return None

def rag_pipeline(query, contents, index, k=5):
    """
    Performs the full Retrieve-and-Generate process.
    """
    if index is None or not contents:
        return "The document has not been processed yet. Please upload a PDF.", []

    # 1. Retrieve: Find relevant content from the document
    query_emb = embedding_model.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_emb)
    _, I = index.search(query_emb, k)
    
    retrieved_contents = [contents[i] for i in I[0] if i < len(contents)]

    if not retrieved_contents:
        return "I couldn't find any relevant information in the document to answer your question.", []

    # 2. Generate: Create a prompt and generate a summarized answer
    context_string = "\n\n---\n\n".join(retrieved_contents)
    prompt = f"""
    You are an expert assistant. Your task is to answer the user's question based *only* on the following context extracted from a document.
    Summarize the information in a clear, narrative paragraph. Do not mention the context directly.
    If the answer is not available in the provided context, state that you cannot find the information in the document.

    CONTEXT:
    {context_string}

    QUESTION: {query}

    ANSWER:
    """.strip()

    # Load model and generate the answer
    tokenizer, model = _load_llm()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=250, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # A simple check to ensure the answer is meaningful
    if not answer or "cannot find the information" in answer.lower():
         return "I couldn't find a direct answer to your question in the provided document.", retrieved_contents

    return answer, retrieved_contents