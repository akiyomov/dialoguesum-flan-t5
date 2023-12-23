from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
import torch
from torch import nn

def load_pretrained_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_summary(model, tokenizer, dialogue):
    prompt = f"""
    Summarize the following conversation.
    
    {dialogue}
    
    Summary: """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids=input_ids, max_new_tokens=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)
