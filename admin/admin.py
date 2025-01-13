import boto3
import streamlit as st
import os
import uuid
import logging
import glob

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 Client
s3_client = boto3.client("s3")

# Set BUCKET_NAME
BUCKET_NAME = "ragchatpdf2"

# Bedrock imports
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Bedrock client
bedrock_client = boto3.client(service_name="bedrock-runtime")

# Embeddings
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock_client
)

# Generate unique ID
def get_unique_id():
    return str(uuid.uuid4())

# Split text into smaller chunks
def split_text(pages, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)

# Create and upload vector store to S3
def create_vector_store(request_id, documents):
    try:
        vector_store = FAISS.from_documents(documents, bedrock_embeddings)
        folder_path = f"/tmp/{request_id}/"
        os.makedirs(folder_path, exist_ok=True)

        # Save vector store locally
        vector_store.save_local(folder_path)

        # Upload FAISS files to S3
        for file in glob.glob(f"{folder_path}*"):
            s3_key = f"vector_stores/{request_id}/{os.path.basename(file)}"
            s3_client.upload_file(file, BUCKET_NAME, s3_key)

        logger.info("Vector store uploaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return False

# Main Streamlit app
def main():
    st.title("Admin Site - Upload PDF and Create Vector Store")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file:
        request_id = get_unique_id()
        saved_file_name = f"/tmp/{request_id}.pdf"

        with open(saved_file_name, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()

        st.write(f"Total Pages: {len(pages)}")

        documents = split_text(pages)
        st.write(f"Total Chunks: {len(documents)}")

        st.write("Creating Vector Store...")
        if create_vector_store(request_id, documents):
            st.success("Vector store uploaded to S3 successfully.")
        else:
            st.error("Failed to upload vector store.")

# Run the app
if __name__ == "__main__":
    main()
