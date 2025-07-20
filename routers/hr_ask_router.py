from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from datetime import date
from starlette.concurrency import run_in_threadpool
import platform

from langchain_teddynote import logging
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS

from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate

from config import OPENAI_API_KEY, LANGCHAIN_PROJECT
from hr_context import fetch_employee_summary, CATEGORIES

# LangSmith 추적 설정
logging.langsmith(project_name=LANGCHAIN_PROJECT)

router = APIRouter()

# 모델 설정
EMBED_MODEL = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
LLM_MODEL = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=OPENAI_API_KEY)

# 날짜 출력 (운영 중 확인용)
today = date.today()
formatted_today = today.strftime("%Y년 %#m월 %#d일" if platform.system() == "Windows" else "%Y년 %-m월 %-d일")
print(formatted_today)

# 출력 모델 정의
class HRAnswer(BaseModel):
    answer: str = Field(description="정중하고 자연스러운 한국어 답변")

qa_parser = JsonOutputParser(pydantic_object=HRAnswer)

# 프롬프트 템플릿
qa_prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""
당신은 사내 인사 시스템 사용자 가이드를 기반으로 질문에 답하는 어시스턴트입니다.

응답 규칙:
1. 모든 답변은 정중하고 자연스러운 한국어로 작성하세요.
2. 답변은 반드시 일반 텍스트 문장만으로 구성하세요 (코드/마크다운 금지).
3. 질문과 관련된 정보가 없으면: "{{질문 키워드}} 관련 정보는 존재하지 않습니다."
4. 사내 인사 시스템과 무관한 질문이면: "해당 질문에 대한 답변은 지원되지 않습니다."
5. 줄바꿈은 반드시 '\\n' 문자열로 표현하세요.
6. 최종 응답은 아래와 같은 JSON 형식 문자열로만 출력하세요 (형식 외 텍스트 없음).

예시:
{{
  "answer": "공지사항을 작성하려면 작성 페이지로 이동하세요.\\n제목과 본문은 필수 항목입니다."
}}

다음은 사원님의 HR 요약 문서입니다. 이를 참고해 아래 질문에 답하세요.

{context}

질문:
{input}
"""
)

# 벡터스토어 생성
def build_vectorstore(text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    return FAISS.from_documents(docs, EMBED_MODEL)

# LangChain QA 체인 구성
def make_qa_chain(vs: FAISS):
    doc_chain = create_stuff_documents_chain(llm=LLM_MODEL, prompt=qa_prompt) | qa_parser
    return create_retrieval_chain(
        retriever=vs.as_retriever(search_kwargs={"k": 4}),
        combine_docs_chain=doc_chain
    )

# 요청 모델
class HRAskRequest(BaseModel):
    employee_id: int
    question: str

# 엔드포인트
@router.post("/ask-hr")
async def ask_hr(req: HRAskRequest):
    hr_text = await run_in_threadpool(
        lambda: fetch_employee_summary(req.employee_id, list(CATEGORIES.keys()))
    )

    if not hr_text.strip():
        raise HTTPException(status_code=404, detail="해당 사원 정보가 없습니다.")

    try:
        vs = build_vectorstore(hr_text)
        qa = make_qa_chain(vs)

        response = qa.invoke({
            "input": req.question,
            "context": hr_text
        }, config={"tags": ["ask-hr", LANGCHAIN_PROJECT]})

        return response["answer"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 처리 중 오류: {str(e)}")