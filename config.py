# config.py
import os
from dotenv import load_dotenv

# .env 파일에서 환경변수 불러오기
load_dotenv()

# ===== DB 연결 설정 =====
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# ===== OpenAI API 키 =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ===== LangChain Tracing 설정 =====
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")