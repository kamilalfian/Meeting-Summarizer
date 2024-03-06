from rouge_score import rouge_scorer
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from transformers import pipeline
from textSummarizer.entity import ModelEvaluationConfig
import torch
import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
hostname = os.getenv("DB_HOSTNAME")
database = os.getenv("DB_DATABASE")
username = os.getenv("DB_USERNAME")
pwd = os.getenv("DB_PWD")
port_id = os.getenv("DB_PORT")
conn= None

try:
    with psycopg2.connect(host = hostname, dbname = database, user = username,
                           password = pwd, port = port_id) as conn:
        with conn.cursor() as cur:
            print('okay')
except Exception as error:
    print(error)
finally:
    if conn is not None:
        conn.close()