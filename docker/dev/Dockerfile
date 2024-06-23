FROM python:3.11
WORKDIR /llm-api
COPY . .

RUN pip install --upgrade pip && \
    pip install fastapi pydantic requests uvicorn python-dotenv openai && \
    pip install -e .

EXPOSE 18950
ENTRYPOINT ["/bin/bash", "-c", "uvicorn src.main:app --host 0.0.0.0 --port 18950"]
