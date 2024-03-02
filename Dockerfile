FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt
RUN pip install --upgrade accelerateg
RUN apt-get update && apt-get install -y cron

CMD ["python", "app.py"]