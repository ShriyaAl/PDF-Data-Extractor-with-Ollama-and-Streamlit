#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip3 install --upgrade --quiet langchain langchain-community chromadb')
get_ipython().system('pip3 install --upgrade --quiet pypdf pandas streamlit sentence-transformers ollama')


# In[4]:


# Import necessary modules
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama  # Replace OpenAI with Ollama

import os
import streamlit as st  
import pandas as pd


# In[5]:


from langchain_community.llms import Ollama

# Initialize FREE LLM (Mistral 7B via Ollama)
llm = Ollama(model="mistral")  # No API key needed!


# In[11]:


# Open terminal and run this in myenv, to run ollama server

# ollama serve
# After that, you pulled the mistral model tou use in your project with ollama pull mistral


# In[10]:


get_ipython().system('ollama --version')


# In[12]:


# Invoke the model to generate a response
response = llm.invoke("Tell me a joke about cats")

print(response)  # Output the joke


# In[14]:


# Processing term sheet pdf

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load PDF document
loader = PyPDFLoader("CIIE Term Sheet Template.pdf")  # Replace with your actual file
pages = loader.load()

# ✅ Step 2: Split into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,   # Each chunk is ~1500 characters
    chunk_overlap=200,  # Overlapping context for better continuity
    length_function=len,
    separators=["\n\n", "\n", " "]  # Try to keep paragraphs intact
)

chunks = text_splitter.split_documents(pages)

# ✅ Step 3: Inspect the Split Chunks
print(f"Total chunks: {len(chunks)}\n")
for i, chunk in enumerate(chunks[:3]):  # Print first 3 chunks
    print(f"Chunk {i+1}:\n{chunk.page_content}\n")
    print("=" * 50)


# In[24]:


# Loading the embedding

# # ✅ Step 1: Import the required modules
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.evaluation import load_evaluator

# # ✅ Step 2: Define the free embedding function
# def get_embedding_function():
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return embeddings

# # ✅ Step 3: Get embeddings
# embedding_function = get_embedding_function()
# test_vector = embedding_function.embed_query("cat")

# # ✅ Step 4: Load the evaluator (measures how similar embeddings are)
# evaluator = load_evaluator(
#     evaluator="embedding_distance", 
#     embeddings=embedding_function
# )

# # ✅ Step 5: Evaluate similarity between prediction & reference
# print(evaluator.evaluate_strings(prediction="Amsterdam", reference="coffeeshop"))
# print(evaluator.evaluate_strings(prediction="Paris", reference="coffeeshop"))

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Load PDF document
loader = PyPDFLoader("CIIE Term Sheet Template.pdf")  # Replace with your actual file
pages = loader.load()

# ✅ Split into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,   
    chunk_overlap=200,  
    length_function=len,
    separators=["\n\n", "\n", " "]  
)

chunks = text_splitter.split_documents(pages)

# ✅ Check chunk count
print(f"Total chunks: {len(chunks)}")


# In[26]:


# Vector database
# import uuid
# from langchain.vectorstores import Chroma

# # ✅ Step 2: Create and Store Vector Embeddings
# def create_vectorstore(chunks, embedding_function, vectorstore_path):
#     # Create unique IDs for each chunk
#     unique_ids = set()
#     unique_chunks = []
    
#     for chunk in chunks:
#         chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.page_content))
#         if chunk_id not in unique_ids:
#             unique_ids.add(chunk_id)
#             unique_chunks.append(chunk)

#     # ✅ Step 3: Store Chunks in ChromaDB
#     vectorstore = Chroma.from_documents(
#         documents=unique_chunks, 
#         embedding=embedding_function,  # Free embeddings
#         persist_directory=vectorstore_path
#     )

#     vectorstore.persist()  # Save data to disk
#     return vectorstore

# # ✅ Step 4: Create the Vectorstore (Saves Chunks for Retrieval)
# vectorstore = create_vectorstore(
#     chunks=chunks, 
#     embedding_function=embedding_function, 
#     vectorstore_path="vectorstore_test"
# )

# print("✅ Vectorstore successfully created & persisted!")

from langchain.vectorstores import Chroma
from langchain.schema import Document

# ✅ Initialize Vector Store
vectorstore = Chroma(
    persist_directory="vectorstore_chroma", 
    embedding_function=embedding_function  # Use the HuggingFace embeddings
)

# ✅ Convert chunks to Chroma format
documents = [Document(page_content=chunk.page_content) for chunk in chunks]

# ✅ Add Documents to Chroma DB
vectorstore.add_documents(documents)
vectorstore.persist()
print(f"Number of documents in vectorstore: {vectorstore._collection.count()}")



# In[18]:


get_ipython().system('pip install -U langchain-chroma')


# In[27]:


# Query for relevant data

# ✅ Create Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",  # ✅ Maximal Marginal Relevance
    search_kwargs={"k": 5}  # ✅ Retrieve top 5 most relevant chunks
)

# ✅ Query for Key Financing Terms
query = "What are the key financing terms in the term sheet?"
relevant_chunks = retriever.invoke(query)

# ✅ Print Retrieved Chunks
for i, chunk in enumerate(relevant_chunks):
    print(f"Chunk {i+1}:\n{chunk.page_content}\n{'='*50}")


# In[28]:


# Prompt for the term sheet

PROMPT_TEMPLATE = """
You are a legal and financial document analysis assistant.  
Your task is to extract **clear, factual, and concise answers** from the provided term sheet.  

Use the following retrieved sections of the document to answer the question.  
If the information is not present, **explicitly state that the term sheet does not provide an answer**.  
**Do not assume, infer, or fabricate any details.**  

{context}  

---

**Question:** {question}  
**Answer based on the term sheet:**  
"""


# In[29]:


# Generating responses

from langchain.prompts import ChatPromptTemplate

# ✅ Concatenate context text with clear separation
context_text = "\n\n---\n\n".join([doc.page_content.strip() for doc in relevant_chunks])

# ✅ Define a specialized prompt for term sheet extraction
PROMPT_TEMPLATE = """
You are a legal and financial document analysis assistant.  
Your task is to extract **clear, factual, and concise answers** from the provided term sheet.  

Use the following retrieved sections of the document to answer the question.  
If the information is not present, **explicitly state that the term sheet does not provide an answer**.  
**Do not assume, infer, or fabricate any details.**  

{context}  

---

**Question:** {question}  
**Answer based on the term sheet:**  
"""

# ✅ Create and format the prompt
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, 
                                question="What are the key financing terms in the term sheet?")

print(prompt)


# In[30]:


# Response Generation

import ollama

# ✅ Use Ollama to generate a response
response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])

# ✅ Print response
print(response["message"]["content"])  # Extracts text response


# In[32]:


# ✅ Define RAG Chain Using LangChain Expression Language (LCEL)

from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# ✅ Function to Format Retrieved Chunks
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# ✅ Define Prompt Template
PROMPT_TEMPLATE = """
You are a legal and financial document analysis assistant.
Use the following retrieved context to answer the question.
If the information is not in the document, say you don't know.

{context}

---

**Question:** {question}
**Answer:**
"""
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),  # Use `RunnableLambda`
        "question": RunnablePassthrough(),
    }
    | prompt_template
    | llm
)

# ✅ Invoke RAG Chain with Ollama
response = rag_chain.invoke("What are the key financing terms in the term sheet?")

# ✅ Print the Response
print(response)


# In[37]:


# Generate Structured Responses

import ollama
import json
import re
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# ✅ Load Ollama Model
llm = Ollama(model="mistral")  # Ensure you're using the correct model

# ✅ Define Dummy Retriever (Replace with your actual retriever)
def dummy_retriever(_):
    return ["Dummy term sheet text for testing."]

retriever = RunnableLambda(dummy_retriever)

# ✅ Function to Format Retrieved Chunks
def format_docs(docs):
    return "\n\n---\n\n".join(docs)

# ✅ Define Prompt Template for Term Sheet Extraction
PROMPT_TEMPLATE = """
You are an expert in analyzing financial term sheets. Extract the key investment details in a structured JSON format.

{context}

---

Extract the following details:
1. **Company Name**
2. **Investor Name**
3. **Type of Security** (e.g., Convertible Preference Shares, Equity)
4. **Pre-money Valuation**
5. **Financing Amount**
6. **Investment Tranches** (Number of stages in which funds are disbursed)
7. **Key Investment Conditions**
8. **Equity Shareholding Information**
9. **Voting Rights & Governance Terms**
10. **Liquidation Preferences**

Provide ONLY the JSON output, without any explanations:

```json
{{
    "company_name": "Extracted Company Name",
    "investor_name": "Extracted Investor Name",
    "security_type": "Extracted Security Type",
    "pre_money_valuation": "Extracted Pre-money Valuation",
    "financing_amount": "Extracted Financing Amount",
    "investment_tranches": "Extracted Investment Tranches",
    "key_conditions": "Extracted Key Investment Conditions",
    "equity_shareholding": "Extracted Equity Shareholding Information",
    "governance_terms": "Extracted Governance Terms",
    "liquidation_preferences": "Extracted Liquidation Preferences"
}}
```
"""
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# ✅ Define RAG Chain
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt_template
    | llm  # ❌ Removed `.with_structured_output()` since it's not supported
)

# ✅ Invoke RAG Chain with Ollama
response = rag_chain.invoke("Extract key financial details from the term sheet.")

# ✅ Extract JSON from Response
json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)

if json_match:
    json_text = json_match.group(1)  # Extract JSON content
    try:
        extracted_data = json.loads(json_text)  # Parse JSON
        print(json.dumps(extracted_data, indent=4))
    except json.JSONDecodeError:
        print("Error: The extracted JSON is not valid.")
        print(json_text)
else:
    print("Error: No valid JSON found in the response.")
    print(response)



# In[43]:


# Output in Dataframes
import json
import re
import pandas as pd
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.output_parsers import OutputFixingParser
from langchain.schema import BaseOutputParser

# ✅ Load Ollama Model
llm = Ollama(model="mistral")  # Ensure correct model

# ✅ Define Dummy Retriever (Replace with actual retriever logic)
def dummy_retriever(_):
    return ["Dummy term sheet text for testing."]

retriever = RunnableLambda(dummy_retriever)

# ✅ Function to Format Retrieved Chunks
def format_docs(docs):
    return "\n\n---\n\n".join(docs)

# ✅ Define Prompt Template
PROMPT_TEMPLATE = """
You are an expert in analyzing financial term sheets. Extract the key investment details in a structured JSON format.

{context}

---

Extract the following details:
- **Company Name**
- **Investor Name**
- **Type of Security** (e.g., Convertible Preference Shares, Equity)
- **Pre-money Valuation**
- **Financing Amount**
- **Investment Tranches** (Number of stages in which funds are disbursed)
- **Key Investment Conditions**
- **Equity Shareholding Information**
- **Voting Rights & Governance Terms**
- **Liquidation Preferences**

Provide ONLY the JSON output, without any explanations.
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# ✅ Define Output Schema
class FinancialTermSheetParser(BaseOutputParser):
    def parse(self, text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse output as JSON.")

output_parser = OutputFixingParser.from_llm(parser=FinancialTermSheetParser(), llm=llm)

# ✅ Define RAG Chain with Forced JSON Output
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt_template
    | llm
    | output_parser  # Forces structured JSON output
)

# ✅ Invoke RAG Chain
response = rag_chain.invoke("Extract key financial details from the term sheet.")

# ✅ Convert to DataFrame
df = pd.DataFrame([response])
print(df)


