from fastapi import FastAPI
from pydantic import BaseModel
import json
from dotenv import load_dotenv
from langchain_teddynote import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# ===== Load environment and init LangSmith =====
load_dotenv()
logging.langsmith("test")  # LangSmith 프로젝트명 등록

# ===== FastAPI App =====
app = FastAPI()

# ===== PDF Load & Vectorstore =====
loader = PyMuPDFLoader("data/사이트_설명서_v2.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
vectorstore = FAISS.from_documents(split_docs, embedding_model)

# ===== LLM & Prompt Chain =====
llm = OllamaLLM(model="mistral")

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant answering questions based on an internal system user guide.

Instructions:
1. Provide a natural and user-friendly explanation in response to the question.
2. Do NOT mention or include the endpoint URL in the answer text itself.
3. If and only if there is a clearly related API endpoint, include it in your JSON under the "endpoint" key; otherwise set "endpoint" to an empty string.
4. Return only a JSON response in the following format. Do not include any text outside the JSON block.
5. The JSON keys must be in English.
6. All answers should be made in Korean.

Format:
{
  "answer": "Your natural language response here.",
  "endpoint": "/api/related-endpoint"  # or "" if there's no relevant endpoint
}

Context:
{context}

Question:
{question}
"""
)


qa_chain = prompt_template | llm

# ===== API Input Model =====
class QueryRequest(BaseModel):
    query: str

# ===== API Endpoint =====
@app.post("/ask")
async def ask_question(query_request: QueryRequest):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"lambda_mult": 0.95, "fetch_k": 4, "k": 3}
    )
    docs = retriever.invoke(query_request.query)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # LangSmith 추적 포함
    response = qa_chain.invoke(
        {"context": context_text, "question": query_request.query},
        config={"tags": ["api_call", "test_project"]}  # LangSmith에서 태깅도 가능
    )

    try:
        parsed = json.loads(response)
        return {
            "question": query_request.query,
            "answer": parsed.get("answer", "잘 모르는 부분입니다, 죄송합니다."),
            "endpoint": parsed.get("endpoint", "제공되는 엔드포인트 없음")
        }
    except json.JSONDecodeError:
        return {
            "question": query_request.query,
            "answer": "The response was not a valid JSON format.",
            "raw_output": response
        }
