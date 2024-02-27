from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline
import torch
from textSummarizer.entity import SummarizerSingleRequest
from typing import Dict, List

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus= AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
        self.summarizer = pipeline("summarization", model=model_pegasus, tokenizer=tokenizer)

    def get_single_prediction(self, feature: SummarizerSingleRequest) -> Dict[str, str]:
        formatted_features = [feature.transcript]
        result = self.summarizer(formatted_features, max_length=600, do_sample=False)
        print({"result": result[0]['summary_text']})
        return ({"result": result[0]['summary_text']})

    def get_batch_prediction(self, features: List[SummarizerSingleRequest]) -> Dict[str, List[str]]:
        formatted_features = [item.transcript for item in features]
        result = self.summarizer(formatted_features, max_length=600, do_sample=False)
        print({"result": [summary['summary_text'] for summary in result]})
        return ({"result": [summary['summary_text'] for summary in result]})