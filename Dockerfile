FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt
RUN pip install --upgrade accelerate
RUN apt-get update && apt-get install -y cron
RUN apt-get install -y nano
ENV EDITOR=nano

CMD ["python", "app.py"]