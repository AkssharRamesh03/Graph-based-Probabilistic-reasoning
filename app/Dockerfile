FROM ubuntu:latest
FROM python:3.10

LABEL authors="aksshar"

ENTRYPOINT ["top", "-b"]

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
