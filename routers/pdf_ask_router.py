from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import json

from langchain_teddynote import logging
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

from config import OPENAI_API_KEY, LANGCHAIN_PROJECT

# ========== APIRouter 생성 ==========
router = APIRouter()

# ========== LangChain 추적 ==========
logging.langsmith(project_name=LANGCHAIN_PROJECT)

# ========== PDF 벡터스토어 초기화 ==========
loader = PyMuPDFLoader("data/사이트_설명서_v2.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY
)
vectorstore = FAISS.from_documents(split_docs, embedding_model)

# ========== LLM & Prompt Chain ==========
llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=OPENAI_API_KEY)

prompt_template = PromptTemplate(
    input_variables=["context", "question", "roles"],
    template="""
You are a helpful assistant answering questions based on an internal system user guide.

Instructions:
1. Provide a natural and user-friendly explanation in response to the question.
2. DO NOT mention or include the endpoint URL in the answer text itself.
3. If and only if there is a clearly related API endpoint, include it in your JSON under the "endpoint" key; otherwise set "endpoint" to an empty string.
4. Return only the raw JSON object. DO NOT include any markdown formatting (such as ```json or ```).
5. DO NOT include any explanatory text before or after the JSON.
6. The JSON keys must be in English.
7. All answers should be written in Korean (한국어).
8. You must only answer questions that are permitted for the roles listed below. If the question cannot be answered due to role limitations, return the following format:

{{
  "answer": "이 질문은 현재 권한으로 열람할 수 없는 내용입니다.",
  "endpoint": ""
}}

User Roles:
{roles}

Context:
{context}

Question:
{question}
"""
)

qa_chain = prompt_template | llm

# ========== 요청 모델 ==========
class QueryRequest(BaseModel):
    query: str
    role: List[str]

# ========== 엔드포인트 ==========
@router.post("/ask")
async def ask_pdf_question(query_request: QueryRequest):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.8}
    )
    docs = retriever.invoke(query_request.query)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # 사용자 권한 정보를 문자열로 변환
    role_text = ", ".join(query_request.role)

    response = qa_chain.invoke(
        {
            "context": context_text,
            "question": query_request.query,
            "roles": role_text
        },
        config={"tags": ["api_call", LANGCHAIN_PROJECT]}
    )

    try:
        parsed = json.loads(response.content)

        if "answer" not in parsed:
            raise ValueError("Missing 'answer' in response")

        return {
            "question": query_request.query,
            "answer": parsed["answer"],
            "endpoint": parsed.get("endpoint", "")
        }

    except json.JSONDecodeError:
        return {
            "question": query_request.query,
            "answer": "응답 형식을 이해하지 못했어요. 다시 질문해 주시겠어요?",
            "raw_output": response.content
        }

    except ValueError as ve:
        return {
            "question": query_request.query,
            "answer": "답변이 정확하지 않아요. 다시 한번 질문을 정리해 주세요.",
            "error_detail": str(ve),
            "raw_output": response.content
        }

    except Exception as e:
        return {
            "question": query_request.query,
            "answer": "예기치 않은 오류가 발생했어요. 관리자에게 문의해 주세요.",
            "error": str(e),
            "raw_output": response.content
        }
