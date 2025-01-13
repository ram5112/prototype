import boto3
import streamlit as st
import os
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


s3_client = boto3.client("s3")


BUCKET_NAME = "ragchatpdf2"


from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS


bedrock_client = boto3.client(service_name="bedrock-runtime")


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


def download_latest_index_files():
    folder_path = "/tmp/"
    os.makedirs(folder_path, exist_ok=True)

    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="vector_stores/")
        files = response.get("Contents", [])

        if not files:
            raise FileNotFoundError("No vector store files found in S3.")

        # Get the latest prefix
        latest_prefix = sorted(files, key=lambda x: x['LastModified'], reverse=True)[0]['Key'].rsplit("/", 1)[0]

        # Download both index.faiss and index.pkl files
        for file_ext in ["index.faiss", "index.pkl"]:
            s3_key = f"{latest_prefix}/{file_ext}"
            local_path = f"{folder_path}{file_ext}"
            s3_client.download_file(BUCKET_NAME, s3_key, local_path)

        logger.info("FAISS index files downloaded successfully.")
        return f"{folder_path}index.faiss", f"{folder_path}index.pkl"

    except Exception as e:
        logger.error(f"Error downloading index files from S3: {e}")
        st.error("Failed to download the latest FAISS index files from S3.")
        return None, None


def load_faiss_index():
    faiss_file, pkl_file = download_latest_index_files()
    if not faiss_file or not pkl_file:
        return None

    try:
        return FAISS.load_local(
            folder_path="/tmp/",
            index_name="index",
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        st.error("Failed to load the FAISS index.")
        return None


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


def main():
    st.title("User Site - Query PDF Knowledge Base")

    st.write("The app will automatically load the latest vector store from S3.")

    st.write("Loading the vector store...")
    vectorstore = load_faiss_index()
    if vectorstore:
        st.success("Vector store loaded successfully.")
    else:
        st.error("Failed to load vector store.")
        return

    question = st.text_input("Ask your question")
    if st.button("Ask"):
        llm = get_llm()
        with st.spinner("Querying..."):
            try:
                response = get_response(llm, vectorstore, question)
                st.write("Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
