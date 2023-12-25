from datasets import load_dataset
from transformers import AutoTokenizer

def load_custom_dataset(dataset_name,tokenizer):
    dataset = load_dataset(dataset_name)
    tokenized_datasets = tokenize_datasets(dataset, tokenizer)
    tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)
    return tokenized_datasets

def tokenize_function(example, tokenizer):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids

    return example

def tokenize_datasets(dataset, tokenizer):
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    return tokenized_datasets.remove_columns(['id', 'topic','dialogue', 'summary'])
