from fastapi import FastAPI
from routers.pdf_ask_router import router as pdf_router
from routers.hr_ask_router import router as hr_router

app = FastAPI()

# 각각 라우터 등록
app.include_router(pdf_router)
app.include_router(hr_router)