from fastapi import APIRouter
from pydantic import BaseModel, Field
import json

from langchain_teddynote import logging
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import OPENAI_API_KEY, LANGCHAIN_PROJECT

# ========= LangSmith 추적 설정 =========
logging.langsmith(project_name=LANGCHAIN_PROJECT)

# ========= 라우터 =========
router = APIRouter()

# ========= 문서 벡터화 =========
loader = PyMuPDFLoader("data/사이트_설명서_v2.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(split_docs, embedding_model)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ========= 출력 모델 및 파서 정의 =========
class QAResponse(BaseModel):
    answer: str = Field(description="한국어로 작성된 자연스러운 설명")
    endpoint: str = Field(description="관련된 API endpoint 경로. 없으면 빈 문자열")

qa_parser = JsonOutputParser(pydantic_object=QAResponse)

# ========= 프롬프트 정의 =========
qa_prompt_template = PromptTemplate(
    input_variables=["context", "input", "roles"],
    template="""
You are a helpful assistant answering questions based on an internal system user guide.

Instructions:
1. If the question is clearly related to the internal guide (Context), answer based on the provided Context.
2. If the question is NOT related to the guide or is a general/common sense question, respond naturally as a chatbot would.
3. DO NOT mention or include the endpoint URL in the answer text itself.
4. If and only if there is a clearly related API endpoint, include it in your JSON under the "endpoint" key; otherwise set "endpoint" to an empty string.
5. Return only the raw JSON object. DO NOT include any markdown formatting (such as ```json or ```).
6. DO NOT include any explanatory text before or after the JSON.
7. The JSON keys must be in English.
8. All answers should be written in Korean (한국어).
9. If the user does not have an appropriate role to access the information, return the following message:
   {{
     "answer": "이 질문은 현재 권한으로 열람할 수 없는 내용입니다.",
     "endpoint": ""
   }}
10. 답변은 최대 300자 이내로 작성하세요. 문장은 간결하고 핵심적인 정보 위주로 정리하세요.

Your response should follow this JSON format:
{{
  "answer": "",
  "endpoint": ""
}}

User Roles:
{roles}

Context:
{context}

Question:
{input}
"""
)

qa_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("history"),
    SystemMessagePromptTemplate(prompt=qa_prompt_template)
])

contextualize_q_system_prompt = """
Given a chat history, the latest user question, employee summary information (included in the first history message), \
and metadata about the HR management system tables, formulate a standalone question. \
The standalone question must be understandable without the chat history and \
should utilize the given employee summary and table metadata if relevant. 

**Do NOT answer the question. Only generate or reformulate the question.**

질문은 반드시 한국어 존댓말로 해줘.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

# ========= LLM 및 체인 구성 =========
llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=OPENAI_API_KEY)

contextual_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt
)

# output parser 연결된 문서 응답 체인
combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt) | qa_parser

rag_chain = create_retrieval_chain(
    retriever=contextual_retriever,
    combine_docs_chain=combine_docs_chain
)

# ========= 세션 관리 =========
chat_history_store = {}

def get_chat_history(session_id: str) -> ChatMessageHistory:
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]

def truncate_history(history: ChatMessageHistory, max_turns=4):
    truncated = history.messages[-(max_turns * 2):]
    history.clear()
    for msg in truncated:
        history.add_message(msg)

chain_with_history = RunnableWithMessageHistory(
    runnable=rag_chain,
    get_session_history=get_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)

# ========= 요청 모델 =========
class QueryRequest(BaseModel):
    query: str
    session_id: int
    roles: list[str]

# ========= 엔드포인트 =========
@router.post("/ask")
async def ask_conversational_question(query_request: QueryRequest):
    history = get_chat_history(query_request.session_id)
    truncate_history(history)

    response = chain_with_history.invoke(
        {
            "input": query_request.query,
            "roles": ", ".join(query_request.roles),
            "context": ""  # context는 retriever 내부에서 자동 주입
        },
        config={
            "tags": ["conversational-rag", LANGCHAIN_PROJECT],
            "configurable": {"session_id": query_request.session_id}
        }
    )

    # 히스토리에 사용자 질문 저장
    history.add_user_message(HumanMessage(content=query_request.query))

    try:
        # 응답 파싱
        if isinstance(response, str):
            parsed = json.loads(response)
        elif isinstance(response, dict):
            parsed = response
        else:
            raise ValueError("Unknown response type")

        # ⬇️ answer가 dict일 수 있으므로 안전하게 처리
        nested = parsed.get("answer")
        print("✅ parsed:", parsed)
        print("✅ nested (parsed['answer']):", nested)

        if isinstance(nested, dict):
            answer_text = nested.get("answer", "")
            endpoint_text = nested.get("endpoint", "")
        elif isinstance(nested, str):
            answer_text = nested
            endpoint_text = parsed.get("endpoint", "")
        else:
            answer_text = str(nested)
            endpoint_text = parsed.get("endpoint", "")

        print("✅ answer_text:", answer_text)
        print("✅ endpoint_text:", endpoint_text)

        history.add_ai_message(AIMessage(content=answer_text))

        # 🔍 업데이트된 히스토리 출력
        print("=== Chat History AFTER invoke ===")
        for msg in history.messages:
            print(f"[{msg.type.upper()}] {msg.content}")
        print("=================================")

        return {
            "answer": answer_text,
            "endpoint": endpoint_text
        }

    except json.JSONDecodeError:
        history.add_ai_message(AIMessage(content="응답 형식을 이해하지 못했어요. 다시 질문해 주시겠어요?"))
        return {
            "question": query_request.query,
            "answer": "응답 형식을 이해하지 못했어요. 다시 질문해 주시겠어요?",
            "raw_output": response
        }

    except Exception as e:
        history.add_ai_message(AIMessage(content="예기치 않은 오류가 발생했어요. 관리자에게 문의해 주세요."))
        return {
            "question": query_request.query,
            "answer": "예기치 않은 오류가 발생했어요. 관리자에게 문의해 주세요.",
            "error": str(e),
            "raw_output": response
        }
