from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from textSummarizer.entity import ModelTrainerConfig
import torch
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not os.path.exists(os.path.join(self.config.root_dir, "tokenizer")) or not os.listdir(os.path.join(self.config.root_dir, "tokenizer")):
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
            print('loading tokenizer from huggingface hub')
        else:
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
            print('loading tokenizer from artifacts')
        if not os.path.exists(os.path.join(self.config.root_dir, "pegasus_x-model")) or not os.listdir(os.path.join(self.config.root_dir, "pegasus_x-model")):
            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
            print('loading model from huggingface hub')
        else:
            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(self.config.root_dir, "pegasus_x-model")).to(device)
            print('loading model from artifacts')
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        # loading data
        dataset_converted = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            metric_for_best_model=self.config.metric_for_best_model, greater_is_better=self.config.greater_is_better,
            per_device_train_batch_size=self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_eval_batch_size, weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy, save_steps=self.config.save_steps, gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )

        trainer = Trainer(model=model_pegasus, args=trainer_args,
                          tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                          train_dataset=dataset_converted["train"],
                          eval_dataset=dataset_converted["validation"])

        trainer.train()

        ## Save model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus_x-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))