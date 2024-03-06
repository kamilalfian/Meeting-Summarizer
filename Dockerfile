FROM python:3.10-slim-buster

# Install PostgreSQL client and Nano
RUN apt-get update && apt-get install -y postgresql postgresql-client nano

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt
RUN pip install --upgrade accelerate

CMD ["python", "app.py"]