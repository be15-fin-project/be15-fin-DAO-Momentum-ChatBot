from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import date
from starlette.concurrency import run_in_threadpool
import platform
import json

from langchain_teddynote import logging
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from config import OPENAI_API_KEY, LANGCHAIN_PROJECT
from hr_context import fetch_employee_summary, CATEGORIES

# LangSmith 추적 설정
logging.langsmith(project_name=LANGCHAIN_PROJECT)

router = APIRouter()

# 모델 및 체인 설정
EMBED_MODEL = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
LLM_MODEL = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=OPENAI_API_KEY)

# 오늘 날짜 출력 (운영 중 로그 확인용)
today = date.today()
if platform.system() == "Windows":
    formatted_today = today.strftime("%Y년 %#m월 %#d일")
else:
    formatted_today = today.strftime("%Y년 %-m월 %-d일")
print(formatted_today)

# 템플릿 설정
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
당신은 사내 인사 시스템 사용자 가이드를 기반으로 질문에 답하는 어시스턴트입니다.

지켜야 할 응답 규칙:
1. 모든 답변은 **정중하고 자연스러운 한국어**로 작성하세요.
2. 답변은 반드시 **일반 텍스트 문장만**으로 구성하세요. (JSON, 코드블록, Markdown 금지)
3. 질문에 해당하는 카테고리가 존재하지만, **해당 카테고리에 정보가 없는 경우**, 다음과 같이 답변하세요:
   - "{{질문 키워드}} 관련 정보는 존재하지 않습니다."
4. 숫자 값이 0.0과 같이 소수점 이하가 0인 경우에는 정수로 표시하고
5. 질문이 사내 인사 시스템과 관련이 없는 경우에는 다음과 같이 답변하세요:
   - "해당 질문에 대한 답변은 지원되지 않습니다."

아래는 사원님의 HR 요약 문서입니다. 이를 참고하여 질문에 답변하세요:

{context}

질문:
{question}
"""
)

# 벡터스토어 및 QA 체인 생성
def build_vectorstore(text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    return FAISS.from_documents(docs, EMBED_MODEL)

def make_qa_chain(vs: FAISS) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=LLM_MODEL,
        retriever=vs.as_retriever(search_kwargs={"k": 4}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

# 요청 모델
class HRAskRequest(BaseModel):
    employee_id: int
    question: str

# 라우터 엔드포인트
@router.post("/ask-hr")
async def ask_hr(req: HRAskRequest):
    hr_text = await run_in_threadpool(
        lambda: fetch_employee_summary(req.employee_id, list(CATEGORIES.keys()))
    )

    print(f"[DEBUG] HR Summary for employee {req.employee_id}:\n{hr_text}")

    if not hr_text.strip():
        raise HTTPException(status_code=404, detail="해당 사원 정보가 없습니다.")

    vs = build_vectorstore(hr_text)
    qa = make_qa_chain(vs)

    try:
        result = qa.invoke(
            {
                "query": req.question,
                "question": req.question,
                "context": hr_text
            },
            config={"tags": ["ask-hr", LANGCHAIN_PROJECT]},
        )
        return {"answer": result.get("result", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 처리 중 오류: {e}")