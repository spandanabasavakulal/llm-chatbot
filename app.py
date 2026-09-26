import os
import time
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RAG LLM Chatbot",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# 2. LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error(
        "GEMINI_API_KEY not found. "
        "Please check your .env file."
    )
    st.stop()


# =========================================================
# 3. GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=api_key
)


# =========================================================
# 4. GEMINI MODELS
# =========================================================

# Primary model + fallback models.
# These are current Gemini API model IDs.

CHAT_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash"
]

# Gemini embedding model
EMBEDDING_MODEL = "gemini-embedding-001"


# =========================================================
# 5. SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are an AI Study Assistant working as a document-based
RAG chatbot.

Your job is to answer questions using ONLY the provided
document context.

Rules:

1. Use the uploaded document context whenever relevant.

2. Do not invent information that is not present in
   the document.

3. If the answer is not present in the document, say:
   "I couldn't find this information in the uploaded document."

4. Explain technical concepts clearly and simply.

5. Use examples only when they are supported by the
   document context.

6. For programming questions, provide correct and
   readable code when the requested information is
   available in the document.

7. Do not claim that information is present if it is
   not present.

8. Answer directly and concisely unless the user asks
   for a detailed explanation.
"""


# =========================================================
# 6. SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0


# =========================================================
# 7. PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):
    """
    Extract text from a text-based PDF.

    OCR is not used in this application.
    """

    reader = PdfReader(uploaded_file)

    pages_text = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:
            page_text = page.extract_text()

        except Exception:
            page_text = None

        if page_text and page_text.strip():

            pages_text.append(
                f"[Page {page_number}]\n{page_text}"
            )

    return "\n\n".join(pages_text)


# =========================================================
# 8. CREATE TEXT CHUNKS
# =========================================================

def create_chunks(
    text,
    chunk_size=1500,
    overlap=300
):
    """
    Split document text into overlapping chunks.
    """

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start += chunk_size - overlap

    return chunks


# =========================================================
# 9. CREATE DOCUMENT EMBEDDINGS
# =========================================================

def create_embeddings(chunks):
    """
    Create Gemini embeddings for document chunks.
    """

    if not chunks:
        return np.array(
            [],
            dtype=np.float32
        )

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=chunks,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768
        )
    )

    embeddings = []

    for embedding in result.embeddings:

        embeddings.append(
            embedding.values
        )

    return np.array(
        embeddings,
        dtype=np.float32
    )


# =========================================================
# 10. CREATE QUERY EMBEDDING
# =========================================================

def create_query_embedding(query):
    """
    Create an embedding for the user's question.
    """

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )
    )

    return np.array(
        result.embeddings[0].values,
        dtype=np.float32
    )


# =========================================================
# 11. RETRIEVE RELEVANT CHUNKS
# =========================================================

def retrieve_relevant_chunks(
    query,
    chunks,
    embeddings,
    top_k=3
):
    """
    Retrieve the most relevant document chunks
    using cosine similarity.
    """

    if not chunks:
        return []

    if embeddings is None:
        return []

    if len(embeddings) == 0:
        return []

    query_embedding = create_query_embedding(
        query
    )

    query_norm = np.linalg.norm(
        query_embedding
    )

    if query_norm == 0:
        return []

    # Normalize query vector
    query_embedding = (
        query_embedding / query_norm
    )

    # Normalize document vectors
    document_norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    normalized_embeddings = (
        embeddings /
        np.maximum(
            document_norms,
            1e-10
        )
    )

    # Cosine similarity
    similarities = np.dot(
        normalized_embeddings,
        query_embedding
    )

    # Make sure top_k isn't larger
    # than number of chunks
    top_k = min(
        top_k,
        len(chunks)
    )

    # Get highest similarity scores
    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "chunk": chunks[index],
            "score": float(
                similarities[index]
            )
        })

    return results


# =========================================================
# 12. GENERATE ANSWER
# =========================================================

def generate_answer(
    question,
    retrieved_chunks
):
    """
    Generate an answer from retrieved document context.

    Handles temporary 503/429 Gemini errors using:
        1. Automatic retries
        2. Exponential backoff
        3. Fallback Gemini models
    """

    # -----------------------------------------------------
    # No relevant chunks
    # -----------------------------------------------------

    if not retrieved_chunks:

        return (
            "I couldn't find this information in the "
            "uploaded document."
        )


    # -----------------------------------------------------
    # Build context
    # -----------------------------------------------------

    context_parts = []

    for i, item in enumerate(
        retrieved_chunks
    ):

        context_parts.append(
            f"""
Document Section {i + 1}

{item["chunk"]}
"""
        )

    context = "\n".join(
        context_parts
    )


    # -----------------------------------------------------
    # Build prompt
    # -----------------------------------------------------

    prompt = f"""
Use the following document context to answer
the user's question.

================ DOCUMENT CONTEXT ================

{context}

================ END DOCUMENT CONTEXT ============

User Question:
{question}

Answer strictly using the document context.
"""


    # -----------------------------------------------------
    # Track last error
    # -----------------------------------------------------

    last_error = None


    # -----------------------------------------------------
    # Try Gemini models
    # -----------------------------------------------------

    for model_name in CHAT_MODELS:

        # Three attempts per model
        for attempt in range(3):

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        max_output_tokens=1500
                    )
                )


                # -------------------------------------------------
                # Check response
                # -------------------------------------------------

                if response.text:

                    return response.text.strip()


                return (
                    "The model did not return a text response."
                )


            except Exception as e:

                last_error = e

                error_text = str(e).lower()


                # -------------------------------------------------
                # Temporary errors
                # -------------------------------------------------

                temporary_error = (
                    "503" in error_text
                    or "unavailable" in error_text
                    or "429" in error_text
                    or "resource exhausted" in error_text
                    or "rate limit" in error_text
                )


                if temporary_error:

                    # ---------------------------------------------
                    # Retry with exponential backoff
                    #
                    # attempt 0 -> 2 seconds
                    # attempt 1 -> 4 seconds
                    # attempt 2 -> move to next model
                    # ---------------------------------------------

                    if attempt < 2:

                        wait_time = 2 ** (
                            attempt + 1
                        )

                        time.sleep(
                            wait_time
                        )

                        continue


                # -------------------------------------------------
                # Non-temporary error
                # or retries exhausted
                # -------------------------------------------------

                break


    # =========================================================
    # ALL MODELS FAILED
    # =========================================================

    return (
        "Gemini is temporarily unavailable after "
        "multiple attempts.\n\n"
        "Please try again shortly.\n\n"
        f"Technical error: {last_error}"
    )


# =========================================================
# 13. HEADER
# =========================================================

st.title(
    "📚 RAG LLM Chatbot"
)

st.caption(
    "Upload a PDF and ask questions about it."
)


# =========================================================
# 14. SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "📄 Upload Document"
    )


    # -----------------------------------------------------
    # PDF uploader
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key=(
            f"pdf_uploader_"
            f"{st.session_state.uploader_version}"
        )
    )


    st.divider()


    # =====================================================
    # CLEAR PDF
    # =====================================================

    if st.button(
        "Clear PDF",
        use_container_width=True
    ):

        st.session_state.chunks = []

        st.session_state.embeddings = None

        st.session_state.pdf_name = None

        st.session_state.messages = []

        st.session_state.uploader_version += 1

        st.rerun()


    # =====================================================
    # PROCESS PDF
    # =====================================================

    if uploaded_file is not None:

        if st.button(
            "Process PDF",
            use_container_width=True
        ):

            try:

                # =============================================
                # STEP 1
                # Extract PDF text
                # =============================================

                with st.spinner(
                    "Reading PDF..."
                ):

                    text = extract_pdf_text(
                        uploaded_file
                    )


                # =============================================
                # STEP 2
                # Check extracted text
                # =============================================

                if not text.strip():

                    st.error(
                        "No readable text was found "
                        "in this PDF."
                    )

                    st.info(
                        "This application currently "
                        "supports text-based PDFs only."
                    )

                else:

                    # =========================================
                    # STEP 3
                    # Create chunks
                    # =========================================

                    chunks = create_chunks(
                        text
                    )

                    st.info(
                        f"Created {len(chunks)} "
                        "text chunks."
                    )


                    # =========================================
                    # STEP 4
                    # Create embeddings
                    # =========================================

                    with st.spinner(
                        "Creating document embeddings..."
                    ):

                        embeddings = create_embeddings(
                            chunks
                        )


                    # =========================================
                    # STEP 5
                    # Store in session state
                    # =========================================

                    st.session_state.chunks = (
                        chunks
                    )

                    st.session_state.embeddings = (
                        embeddings
                    )

                    st.session_state.pdf_name = (
                        uploaded_file.name
                    )

                    # Clear previous chat
                    st.session_state.messages = []


                    # =========================================
                    # SUCCESS
                    # =========================================

                    st.success(
                        "PDF processed successfully!"
                    )


            except Exception as e:

                st.error(
                    f"Error processing PDF: {e}"
                )


    # =====================================================
    # PDF STATUS
    # =====================================================

    if st.session_state.pdf_name:

        st.success(
            f"Loaded: "
            f"{st.session_state.pdf_name}"
        )

        st.write(
            f"Chunks: "
            f"{len(st.session_state.chunks)}"
        )


# =========================================================
# 15. DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# 16. CHAT INPUT
# =========================================================

user_input = st.chat_input(
    "Ask something about your PDF..."
)


# =========================================================
# 17. PROCESS USER QUESTION
# =========================================================

if user_input:

    # -----------------------------------------------------
    # Display user message
    # -----------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_input
        )


    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })


    # =====================================================
    # CHECK WHETHER PDF HAS BEEN PROCESSED
    # =====================================================

    if (
        not st.session_state.chunks
        or
        st.session_state.embeddings is None
    ):

        answer = (
            "Please upload and process a PDF first."
        )


    else:

        try:

            # =============================================
            # STEP 1
            # Retrieve relevant chunks
            # =============================================

            with st.spinner(
                "Searching your document..."
            ):

                retrieved_chunks = (
                    retrieve_relevant_chunks(
                        user_input,
                        st.session_state.chunks,
                        st.session_state.embeddings,
                        top_k=3
                    )
                )


            # =============================================
            # STEP 2
            # Generate answer
            # =============================================

            with st.spinner(
                "Generating answer..."
            ):

                answer = generate_answer(
                    user_input,
                    retrieved_chunks
                )


        except Exception as e:

            answer = (
                f"RAG error: {e}"
            )


    # =====================================================
    # DISPLAY ANSWER
    # =====================================================

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            answer
        )


    # =====================================================
    # SAVE ANSWER
    # =====================================================

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# =========================================================
# 18. CLEAR CHAT
# =========================================================

if st.session_state.messages:

    if st.button(
        "Clear Chat"
    ):

        st.session_state.messages = []

        st.rerun()