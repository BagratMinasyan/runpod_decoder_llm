from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

hf_token = os.getenv("RUNPOD_SECRET_hf_key")
print(f"🔑 HF Token available: {'Yes' if hf_token else 'No'}")
print(f"hf token: {hf_token}")

def resolve_dtype(dtype_str: str):
    print(f"🔢 Resolving dtype: {dtype_str}")
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto"
    }
    resolved = mapping.get(dtype_str.lower(), torch.float16)
    print(f"🔢 Resolved to: {resolved}")
    return resolved

def load_model(model_name: str, torch_dtype_str: str = "float16"):
    print(f"🚀 Starting model loading: {model_name}")
    print(f"🔢 Requested dtype: {torch_dtype_str}")
    
    dtype = resolve_dtype(torch_dtype_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Target device: {device}")
    
    if device == "cuda":
        print(f"🔥 CUDA available: {torch.cuda.is_available()}")
        print(f"🔥 CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"🔥 CUDA device name: {torch.cuda.get_device_name(0)}")

    try:
        print("🔤 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token
        )
        print("✅ Tokenizer loaded successfully")
        print(f"📊 Vocab size: {tokenizer.vocab_size}")
        print(f"🔚 EOS token: {tokenizer.eos_token}")
        print(f"🔚 EOS token ID: {tokenizer.eos_token_id}")
        print(f"📝 Pad token: {tokenizer.pad_token}")
        print(f"📝 Pad token ID: {tokenizer.pad_token_id}")
        
    except Exception as e:
        print(f"💥 Tokenizer loading failed: {type(e).__name__} - {str(e)}")
        raise RuntimeError(f"Tokenizer load failed: {type(e).__name__} - {str(e)}")

    try:
        print("🤖 Loading model...")
        print(f"🔢 Using dtype: {dtype}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            token=hf_token
        )
        print("✅ Model loaded from pretrained")
        
        print(f"🖥️ Moving model to device: {device}")
        model = model.to(device).eval()
        print("✅ Model moved to device and set to eval mode")
        
        print(f"📊 Model config: {model.config}")
        print(f"📊 Model dtype: {next(model.parameters()).dtype}")
        print(f"📊 Model device: {next(model.parameters()).device}")
        
        # Print memory usage if CUDA
        if device == "cuda" and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            print(f"🔥 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        
    except Exception as e:
        print(f"💥 Model loading failed: {type(e).__name__} - {str(e)}")
        raise RuntimeError(f"Model load failed: {type(e).__name__} - {str(e)}")

    print("🎉 Model and tokenizer loading completed successfully")
    return model, tokenizer