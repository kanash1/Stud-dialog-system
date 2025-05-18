FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
 && pip install --no-cache-dir -r requirements.txt

COPY wait-for-it.sh .
COPY app/ ./app/

RUN chmod +x wait-for-it.sh

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

