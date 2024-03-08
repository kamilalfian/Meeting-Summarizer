FROM python:3.10-slim-buster

# Install PostgreSQL and Nano
RUN apt update && apt upgrade -y && apt install -y postgresql postgresql-contrib nano

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt
RUN pip install --upgrade accelerate

CMD ["python", "app.py"]