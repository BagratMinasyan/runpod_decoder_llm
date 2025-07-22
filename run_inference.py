from transformers import GenerationConfig
from model_loader import load_model
import torch

def run_inference(prompt, model_name: str, generation_params: dict = None, torch_dtype: str = "float16"):
    try:
        model, tokenizer = load_model(model_name, torch_dtype)

        if tokenizer.eos_token_id is None and hasattr(model.config, "eos_token_id"):
            tokenizer.eos_token_id = model.config.eos_token_id
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = tokenizer.pad_token_id or 2  # 2 is a common eos id in LLaMA

    except Exception as e:
        return {"error": f"Failed to load model/tokenizer: {type(e).__name__} - {str(e)}"}

    try:
        device = model.device
        if isinstance(prompt, str):
            prompt = [prompt]

        max_len = getattr(model.config, "max_position_embeddings", 2048)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception as e:
        return {"error": f"Tokenization error: {type(e).__name__} - {str(e)}"}

    try:
        if not generation_params:
            generation_config = GenerationConfig(do_sample=False, max_new_tokens=256)
        else:
            generation_params.setdefault("max_new_tokens", 256)
            generation_config = GenerationConfig(**generation_params)
    except Exception as e:
        return {"error": f"Invalid generation_config: {type(e).__name__} - {str(e)}"}

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = [tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
        return {
            "results": [
                {"prompt": p, "output": o} for p, o in zip(prompt, decoded)
            ]
        }

    except Exception as e:
        return {"error": f"Generation error: {type(e).__name__} - {str(e)}"}
