from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

hf_token = os.getenv("RUNPOD_SECRET_hf_key")

def resolve_dtype(dtype_str: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto"
    }
    return mapping.get(dtype_str.lower(), torch.float16)

def load_model(model_name: str, torch_dtype_str: str = "float16"):
    dtype = resolve_dtype(torch_dtype_str)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token
        )
    except Exception as e:
        raise RuntimeError(f"Tokenizer load failed: {type(e).__name__} - {str(e)}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            token=hf_token
        ).eval()
    except Exception as e:
        raise RuntimeError(f"Model load failed: {type(e).__name__} - {str(e)}")

    return model, tokenizer
