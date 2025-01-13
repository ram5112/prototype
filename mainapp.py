import streamlit as st
import os
import logging
import boto3
import uuid
import glob
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 Client
s3_client = boto3.client("s3")
bedrock_client = boto3.client(service_name="bedrock-runtime")
BUCKET_NAME = "ragchatpdf2"

# Bedrock Embeddings and LLM
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock_client
)

def get_llm():
    return Bedrock(
        model_id="anthropic.claude-v2:1",
        client=bedrock_client,
        model_kwargs={"max_tokens_to_sample": 512}
    )

# Helper function: Generate unique ID
def get_unique_id():
    return str(uuid.uuid4())

# Helper function: Split text into chunks
def split_text(pages, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)

# Helper function: Create and upload vector store to S3
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

# Helper function: Load FAISS index from S3
def load_faiss_index():
    folder_path = "/tmp/"
    os.makedirs(folder_path, exist_ok=True)

    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="vector_stores/")
        files = response.get("Contents", [])
        if not files:
            raise FileNotFoundError("No vector store files found in S3.")

        latest_prefix = sorted(files, key=lambda x: x['LastModified'], reverse=True)[0]['Key'].rsplit("/", 1)[0]

        for file_ext in ["index.faiss", "index.pkl"]:
            s3_key = f"{latest_prefix}/{file_ext}"
            local_path = f"{folder_path}{file_ext}"
            s3_client.download_file(BUCKET_NAME, s3_key, local_path)

        logger.info("FAISS index files downloaded successfully.")
        return FAISS.load_local(folder_path=folder_path, index_name="index", embeddings=bedrock_embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Error downloading index files from S3: {e}")
        return None

# Helper function: Get chatbot response
def get_response(llm, vectorstore, question):
    prompt_template = """
    Human: Please use the given context to answer the question concisely.
    <context>
    {context}
    </context>
    Question: {question}
    Assistant:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

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
    st.sidebar.title("📄 Admin Panel")
    page = st.sidebar.selectbox("Select Page", ["Chatbot", "Admin"])

    if page == "Admin":
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

    elif page == "Chatbot":
        st.title("💬 PDF Chatbot")
        vectorstore = load_faiss_index()
        if vectorstore:
            st.success("Vector store loaded successfully.")
        else:
            st.error("Failed to load vector store.")
            return

        if "history" not in st.session_state:
            st.session_state["history"] = []

        question = st.text_input("Ask your question:")
        if st.button("Send"):
            llm = get_llm()
            response = get_response(llm, vectorstore, question)
            st.session_state["history"].append((question, response))

        st.write("### Chat History")
        for idx, (q, r) in enumerate(st.session_state["history"]):
            st.write(f"**You:** {q}")
            st.write(f"**Bot:** {r}")

# Run the app
if __name__ == "__main__":
    main()
