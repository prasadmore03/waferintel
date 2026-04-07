FROM python:3.11-slim

WORKDIR /app

COPY requirements_rag.txt .
RUN pip install --no-cache-dir -r requirements_rag.txt

COPY . .

EXPOSE 8502

CMD ["python", "run_rag_app.py"]
