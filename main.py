from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.pdf_ask_router import router as pdf_router
from routers.hr_ask_router import router as hr_router

app = FastAPI()

# CORS 설정 (모든 origin 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_methods=["POST"],  
)

# /api/backend-chatbot/v1로 시작하는 라우터로 통합 등록
app.include_router(pdf_router, prefix="/api/backend-chatbot/v1")
app.include_router(hr_router, prefix="/api/backend-chatbot/v1")

# 헬스 체크
@app.get("/health")
def health_check():
    return {"status": "ok"}
