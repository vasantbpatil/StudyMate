import streamlit as st
import os
from core_rag import parse_pdf_to_contents, create_faiss_index, rag_pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="StudyMate: PDF Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 StudyMate: PDF Q&A with Hugging Face")
st.markdown("Upload your PDF document and get summarized answers to your questions based on its content.")

# --- Session State Initialization ---
# To store data across user interactions
if 'contents' not in st.session_state:
    st.session_state.contents = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

# --- File Uploader and Processing ---
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # Process the file only if it's a new file
    if uploaded_file.name != st.session_state.last_uploaded_file:
        st.session_state.last_uploaded_file = uploaded_file.name
        
        # Save the uploaded file temporarily to a path
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.status("Analyzing your document...", expanded=True) as status:
            try:
                status.write("Parsing document into readable contents...")
                st.session_state.contents = parse_pdf_to_contents(temp_file_path)
                
                if st.session_state.contents:
                    status.write("Creating a searchable index of the content...")
                    st.session_state.index = create_faiss_index(st.session_state.contents)
                    status.update(label="Document processed! You can now ask questions.", state="complete", expanded=False)
                else:
                    st.error("Could not extract any content from the PDF. Please check the file.")
                    st.session_state.index = None
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.session_state.index = None
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

# --- Q&A Section ---
if st.session_state.index:
    st.header("Ask a Question About the PDF Content:")
    query = st.text_input("Enter your question here:", label_visibility="collapsed")
    
    if st.button("Get Answer"):
        if query:
            with st.spinner("Searching for answers..."):
                # Run the RAG pipeline to get the answer and sources
                answer, retrieved_contents = rag_pipeline(
                    query, st.session_state.contents, st.session_state.index
                )

                st.subheader("Answer:")
                st.markdown(f"> {answer}")

                # Display the source paragraphs that the answer was based on
                if retrieved_contents:
                    st.subheader("Relevant Content from the Document:")
                    with st.expander("Click to view the source paragraphs"):
                        for i, content in enumerate(retrieved_contents):
                            st.info(f"**Snippet from document:**\n\n---\n\n{content}")
        else:
            st.warning("Please enter a question to get an answer.")