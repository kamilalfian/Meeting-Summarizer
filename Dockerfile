FROM python:3.10-slim-buster as python-base

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt
RUN pip install --upgrade accelerate

CMD ["python", "app.py"]