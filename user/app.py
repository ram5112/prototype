import boto3
import streamlit as st
import os
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 Client
s3_client = boto3.client("s3")

# Set BUCKET_NAME
BUCKET_NAME = "ragchatpdf2"

# Bedrock imports
from langchain_aws.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

# Bedrock client
bedrock_client = boto3.client(service_name="bedrock-runtime")

# Embeddings
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock_client
)

# Initialize Bedrock LLM
def get_llm():
    return Bedrock(
        model_id="anthropic.claude-v2:1",
        client=bedrock_client,
        model_kwargs={"max_tokens_to_sample": 512}
    )

# Download the latest FAISS files from S3
def download_latest_index_files():
    folder_path = "/tmp/"
    os.makedirs(folder_path, exist_ok=True)

    try:
        # List S3 objects in the vector_stores folder
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="vector_stores/")
        files = response.get("Contents", [])

        # Find the latest .faiss and .pkl files
        faiss_file = None
        pkl_file = None

        for file in files:
            if file["Key"].endswith(".faiss"):
                faiss_file = file["Key"]
            elif file["Key"].endswith(".pkl"):
                pkl_file = file["Key"]

        if not faiss_file or not pkl_file:
            raise FileNotFoundError("FAISS index files (.faiss and .pkl) are not found in the S3 bucket.")

        # Download files to the /tmp/ directory
        faiss_local_path = f"{folder_path}{os.path.basename(faiss_file)}"
        pkl_local_path = f"{folder_path}{os.path.basename(pkl_file)}"

        s3_client.download_file(Bucket=BUCKET_NAME, Key=faiss_file, Filename=faiss_local_path)
        s3_client.download_file(Bucket=BUCKET_NAME, Key=pkl_file, Filename=pkl_local_path)

        logger.info("FAISS index files downloaded successfully.")
        return faiss_local_path, pkl_local_path

    except Exception as e:
        logger.error(f"Error downloading index files from S3: {e}")
        st.error("Failed to download the latest FAISS index files from S3.")
        return None, None

# Load FAISS index
def load_faiss_index():
    faiss_file, pkl_file = download_latest_index_files()
    if not faiss_file or not pkl_file:
        return None

    try:
        return FAISS.load_local(
            index_name=os.path.splitext(os.path.basename(faiss_file))[0],
            folder_path="/tmp/",
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        st.error("Failed to load the FAISS index.")
        return None

# Get a response from the vector store
def get_response(llm, vectorstore, question):
    prompt_template = """
    Human: Please use the given context to answer the question concisely.
    <context>
    {context}
    </context>
    Question: {question}
    Assistant:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = qa({"query": question})
    return result['result']

# Main Streamlit app
def main():
    st.title("User Site - Query PDF Knowledge Base")

    st.write("The app will automatically load the latest vector store from S3.")

    # Load the FAISS index
    st.write("Loading the vector store...")
    vectorstore = load_faiss_index()
    if vectorstore:
        st.success("Vector store loaded successfully.")
    else:
        st.error("Failed to load vector store.")
        return

    # Ask a question
    question = st.text_input("Ask your question")
    if st.button("Ask"):
        llm = get_llm()
        with st.spinner("Querying..."):
            try:
                response = get_response(llm, vectorstore, question)
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")

# Run the app
if __name__ == "__main__":
    main()
