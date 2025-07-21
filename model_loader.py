import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Your static model
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
hf_token = os.getenv("RUNPOD_SECRET_hf_key")

def resolve_dtype(dtype_str: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto"
    }
    return mapping.get(dtype_str.lower(), torch.float16)

def load_model(torch_dtype_str: str = "float16"):
    dtype = resolve_dtype(torch_dtype_str)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=hf_token
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
        token=hf_token
    ).eval()

    return model, tokenizer
