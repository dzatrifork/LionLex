# syntax=docker/dockerfile:1
FROM python:3.11
LABEL authors="David"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

EXPOSE 8000

CMD chainlit run -h docs.py