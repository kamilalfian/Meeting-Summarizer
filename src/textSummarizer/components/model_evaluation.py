from rouge_score import rouge_scorer
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from transformers import pipeline
from textSummarizer.entity import ModelEvaluationConfig
import torch
import pandas as pd


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
        summarizer = pipeline("summarization", model=model_pegasus, tokenizer=tokenizer)
        # loading data
        dataset_eval = load_from_disk(self.config.data_path)
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        # Initialize lists to store ROUGE scores
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        rougeLSUM_scores = []
        # Iterate over the first 60 examples in the test dataset
        for idx, (summary, transcript) in tqdm(
                enumerate(zip(dataset_eval['test']['summary'][:60], dataset_eval['test']['transcript'][:60])),total=60):
            # Generate summary for the example using your model
            prediction = summarizer(transcript, max_length=600, do_sample=False)[0]['summary_text']
            # Calculate ROUGE scores
            scores = scorer.score(summary, prediction)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
            # Calculate ROUGE-LSUM score
            summary_length = len(summary.split())
            prediction_length = len(prediction.split())
            rouge_lsum_score = min(1, prediction_length / summary_length)
            rougeLSUM_scores.append(rouge_lsum_score)
        # Calculate average ROUGE scores
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
        avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
        avg_rougeLSUM = sum(rougeLSUM_scores) / len(rougeLSUM_scores)
        #Convert to csv file
        df = pd.DataFrame({
            'avg_rouge1': [avg_rouge1],
            'avg_rouge2': [avg_rouge2],
            'avg_rougeL': [avg_rougeL],
            'avg_rougeLSUM': [avg_rougeLSUM]
        }, index=['pegasus'])
        df.to_csv(self.config.metric_file_name, index=False)