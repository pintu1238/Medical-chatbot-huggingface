# import os
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# # step1:- Setup LLM with huggingface
# HP_TOKEN=os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

# def load_llm(HUGGINGFACE_REPO_ID):
#   llm=HuggingFaceEndpoint(
#   repo_id=HUGGINGFACE_REPO_ID,
#   temperature=0.5,
#   model_kwargs={"token":HF_TOKEN,
#                 "max_length":"512"}
#   )
#   return llm

# # step2:- connect llm with faiss and create chain

# CUSTOM_PROMPT_TEMPLATE = """
# Use the pieces of information provided in the context to answer user's question.
# If you dont know the answer, just say that you dont know, dont try to make up an answer. 
# Dont provide anything out of the given context

# Context: {context}
# Question: {question}

# Start the answer directly. No small talk please.
# """

# def set_custom_prompt(custome_prompt_template):
#   prompt=PromptTemplate(template=custome_prompt_template, input_variable=["context", "question"])
#   return prompt

# # Load database
# DB_FAISS_PATH="vectorstore/db_faiss"
# embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


# # create qa chain
# qa_chain=RetrievalQA.from_chain_type(
#   llm=load_llm(HUGGINGFACE_REPO_ID),
#   chain_type="stuff",
#   retriever=db.as_retriever(search_kwargs={'k':3}),
#   return_source_documents=True,
#   chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# )

# # Now invoke with a single query
# user_query=input("Write Query Here: ")
# response=qa_chain.invoke({'query': user_query})
# print("RESULT: ", response["result"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])


from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
print("Token Found:", HF_TOKEN is not None)


import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# =========================
# STEP 1: Setup HF Token
# =========================

HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found. Please set your Hugging Face token.")

# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HUGGINGFACE_REPO_ID = "google/flan-t5-large"


# =========================
# STEP 2: Load LLM
# =========================

def load_llm(repo_id):

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=512
    )

    return llm


# =========================
# STEP 3: Custom Prompt
# =========================

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.

If you don't know the answer, just say that you don't know.
Don't try to make up an answer.

Only answer from the provided context.

Context:
{context}

Question:
{question}

Start the answer directly.
"""


def set_custom_prompt(custom_prompt_template):

    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

    return prompt


# =========================
# STEP 4: Load FAISS DB
# =========================

DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)


# =========================
# STEP 5: Create QA Chain
# =========================

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    }
)


# =========================
# STEP 6: Ask Question
# =========================

user_query = input("Write Query Here: ")

response = qa_chain.invoke(
    {"query": user_query}
)

print("\nRESULT:\n")
print(response["result"])

print("\nSOURCE DOCUMENTS:\n")
print(response["source_documents"])