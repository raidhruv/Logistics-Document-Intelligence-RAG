import streamlit as st
import os
from src.build_index import build_vector_store
from src.rag_pipeline import answer_query

st.set_page_config(
    page_title="Logistics Document Intelligence",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Logistics Document Intelligence RAG")
st.markdown("Upload a logistics/compliance PDF and ask questions.")

if "indexed" not in st.session_state:
    st.session_state.indexed = False

# ---------------------------
# PDF Upload Section
# ---------------------------

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None and not st.session_state.get("indexed", False):

    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        with st.spinner("Indexing document..."):
            build_vector_store(file_path)

        st.session_state.indexed = True
        st.success("Document indexed successfully.")

    except Exception as e:
        st.error(f"Indexing failed: {e}")
        st.session_state.indexed = False

# ---------------------------
# Query Section
# ---------------------------

if st.session_state.indexed:

    query = st.text_input("Enter your question")

    if query:
       with st.spinner("Generating answer..."):
        try:
            response = answer_query(query)

            st.subheader("Answer")
            st.write(response["answer"])

            # Clamp confidence between 0 and 1
            confidence = max(0.0, min(1.0, response["confidence"]))
            confidence_percent = confidence * 100

            col1, col2 = st.columns([4, 1])

            with col1:
                st.subheader("Confidence Score")

            with col2:
                st.subheader(f"{confidence_percent:.1f}%")

            st.progress(confidence)

            if response["citations"]:
                st.subheader("Sources")

                for i, chunk in enumerate(response["citations"], 1):
                    with st.expander(f"Chunk {i} — Score: {chunk['score']:.3f}"):
                      text = chunk["metadata"]["text"]
                      st.write(text if len(text) < 800 else text[:800] + "...")


        except Exception as e:
            st.error(f"Error generating answer: {e}")


        #st.subheader("Citations")
        #for i, citation in enumerate(response["citations"], 1):
            #with st.expander(f"Source {i}"):
                #st.write(citation)

else:
    st.info("Upload a document to begin.")
