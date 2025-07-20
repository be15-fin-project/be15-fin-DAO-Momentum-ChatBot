FROM python:3.11-slim

# 필수 패키지 설치 (mariadb 포함)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libmariadb-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# pip 수동 활성화 및 업그레이드
RUN python -m ensurepip && pip install --upgrade pip

# FastAPI와 uvicorn을 전역 pip로 직접 설치
RUN pip install --no-cache-dir fastapi uvicorn[standard]

# Poetry 설치
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# 작업 디렉토리 설정
WORKDIR /app

# Poetry 설정 및 종속성 설치
COPY pyproject.toml poetry.lock* /app/
RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi

# 소스코드 복사
COPY . /app

# FastAPI 실행
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
