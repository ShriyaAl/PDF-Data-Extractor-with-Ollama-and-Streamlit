import streamlit as st
import json
import re
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from PyPDF2 import PdfReader

# ✅ Load LLM (Make sure Ollama is running)
llm = Ollama(model="mistral")

# ✅ Extract text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# ✅ Define Prompt
PROMPT_TEMPLATE = """
You are an expert in analyzing financial documents. Extract key financial details in JSON format.

{text}

---

Extract these details:
1. **Company Name**
2. **Investor Name**
3. **Type of Security**
4. **Pre-money Valuation**
5. **Financing Amount**
6. **Investment Tranches**
7. **Key Investment Conditions**
8. **Equity Shareholding Information**
9. **Governance Terms**
10. **Liquidation Preferences**

Provide ONLY JSON output:

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
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# ✅ Function to process text with LLM
def process_text_with_llm(text):
    rag_chain = prompt_template | llm
    response = rag_chain.invoke({"text": text})
    
    json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
    if json_match:
        json_text = json_match.group(1)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format"}
    return {"error": "No valid JSON found"}

# ✅ Streamlit App
st.title("📄 Term Sheet Data Extractor")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    st.info("Processing file...")
    pdf_text = extract_text_from_pdf(uploaded_file)
    extracted_data = process_text_with_llm(pdf_text)

    if "error" in extracted_data:
        st.error(extracted_data["error"])
    else:
        st.success("Extracted Data:")
        st.json(extracted_data)  # Display structured data

