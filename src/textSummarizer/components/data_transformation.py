import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk
from textSummarizer.entity import DataTransformationConfig
import pandas as pd
from datasets import DatasetDict, Dataset

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features_pure(self, example_batch):
        input_encodings = self.tokenizer(example_batch['transcript'])

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'])

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(example_batch['transcript'], max_length=4096, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], max_length=600, truncation=True)

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        dataset = load_from_disk(self.config.data_path)
        dataset_converted= dataset.map(self.convert_examples_to_features_pure, batched=True)
        # Extract data from the dataset
        train_data = dataset_converted['train']
        validation_data = dataset_converted['validation']
        test_data = dataset_converted['test']
        # Convert to pandas DataFrame
        train_df = pd.DataFrame(train_data)
        validation_df = pd.DataFrame(validation_data)
        test_df = pd.DataFrame(test_data)
        train_df['input_token_length'] = train_df['input_ids'].apply(lambda x: len(x))
        train_df['output_token_length'] = train_df['labels'].apply(lambda x: len(x))
        validation_df['input_token_length'] = validation_df['input_ids'].apply(lambda x: len(x))
        validation_df['output_token_length'] = validation_df['labels'].apply(lambda x: len(x))
        test_df['input_token_length'] = test_df['input_ids'].apply(lambda x: len(x))
        test_df['output_token_length'] = test_df['labels'].apply(lambda x: len(x))
        # Drop rows with input_token_length > 4096
        train_df = train_df[train_df['input_token_length'] <= 4096]
        # Drop rows with input_token_length > 4096
        validation_df = validation_df[validation_df['input_token_length'] <= 4096]
        # Drop rows with input_token_length > 4096
        test_df = test_df[test_df['input_token_length'] <= 4096]
        # Subset train_df to contain 3000 rows
        train_df = train_df.head(3000)
        # Subset validation_df to contain 400 rows
        validation_df = validation_df.head(400)
        # Subset test_df to contain 400 rows
        test_df = test_df.head(400)
        # Convert them back into dataset format
        train_dataset = Dataset.from_pandas(train_df)
        validation_dataset = Dataset.from_pandas(validation_df)
        test_dataset = Dataset.from_pandas(test_df)
        # Put them back into DatasetDict
        dataset = DatasetDict({
            'train': train_dataset,
            'validation': validation_dataset,
            'test': test_dataset})
        dataset_converted = dataset.map(self.convert_examples_to_features, batched=True, remove_columns=['id', 'uid','input_token_length', 'output_token_length'])
        dataset_converted.save_to_disk(os.path.join(self.config.root_dir, "dataset"))