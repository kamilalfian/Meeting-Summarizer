FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt
RUN pip install --upgrade accelerate
RUN apt-get update && apt-get install -y cron
RUN apt-get install -y nano
ENV EDITOR=nano

# Copy your cron job file into the appropriate directory
COPY mycronjob /etc/cron.d/mycronjob

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/mycronjob

# Apply the cron job
RUN crontab /etc/cron.d/mycronjob

# Start the cron service
CMD cron -f && python main.py
