from transformers import GenerationConfig
from model_loader import load_model
import torch

def run_inference(prompt, model_name: str, generation_params: dict = None, torch_dtype: str = "float16"):
    print("🔍 Starting run_inference...")
    print(f"📝 Prompt length: {len(prompt) if isinstance(prompt, str) else len(prompt) if isinstance(prompt, list) else 'unknown'}")
    print(f"🤖 Model: {model_name}")
    print(f"🔢 Dtype: {torch_dtype}")
    print(f"⚙️ Generation params: {generation_params}")
    
    try:
        print("📦 Loading model and tokenizer...")
        model, tokenizer = load_model(model_name, torch_dtype)
        print("✅ Model and tokenizer loaded successfully")

        print("🔧 Setting up EOS token...")
        if tokenizer.eos_token_id is None and hasattr(model.config, "eos_token_id"):
            tokenizer.eos_token_id = model.config.eos_token_id
            print(f"🔧 Set EOS token from model config: {tokenizer.eos_token_id}")
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = tokenizer.pad_token_id or 2
            print(f"🔧 Set EOS token fallback: {tokenizer.eos_token_id}")
        print(f"✅ EOS token ID: {tokenizer.eos_token_id}")

    except Exception as e:
        print(f"💥 Model loading failed: {type(e).__name__} - {str(e)}")
        return {"error": f"Failed to load model/tokenizer: {type(e).__name__} - {str(e)}"}

    try:
        print("🖥️ Setting up device and tokenization...")
        device = model.device
        print(f"🖥️ Using device: {device}")
        
        if isinstance(prompt, str):
            prompt = [prompt]
            print("📝 Converted single prompt to list")
        
        print(f"📝 Processing {len(prompt)} prompt(s)")

        max_len = getattr(model.config, "max_position_embeddings", 2048)
        print(f"📏 Max length: {max_len}")
        
        print("🔤 Tokenizing inputs...")
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        print(f"🔤 Input IDs shape: {inputs['input_ids'].shape}")
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print("✅ Inputs moved to device")
        
    except Exception as e:
        print(f"💥 Tokenization failed: {type(e).__name__} - {str(e)}")
        return {"error": f"Tokenization error: {type(e).__name__} - {str(e)}"}

    try:
        print("⚙️ Setting up generation config...")
        if not generation_params:
            generation_config = GenerationConfig(do_sample=False, max_new_tokens=256)
            print("⚙️ Using default generation config")
        else:
            generation_params.setdefault("max_new_tokens", 256)
            generation_config = GenerationConfig(**generation_params)
            print(f"⚙️ Using custom generation config: {generation_params}")
        
    except Exception as e:
        print(f"💥 Generation config failed: {type(e).__name__} - {str(e)}")
        return {"error": f"Invalid generation_config: {type(e).__name__} - {str(e)}"}

    try:
        print("🚀 Starting text generation...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        print(f"✅ Generation completed. Output shape: {outputs.sequences.shape}")
        
        print("🔤 Decoding outputs...")
        decoded = [tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
        print(f"✅ Decoded {len(decoded)} sequences")
        
        result = {
            "results": [
                {"prompt": p, "output": o} for p, o in zip(prompt, decoded)
            ]
        }
        
        print("🎉 Inference completed successfully")
        print(f"📊 Results count: {len(result['results'])}")
        
        return result

    except Exception as e:
        print(f"💥 Generation failed: {type(e).__name__} - {str(e)}")
        return {"error": f"Generation error: {type(e).__name__} - {str(e)}"}