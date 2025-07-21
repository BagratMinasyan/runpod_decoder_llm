from transformers import GenerationConfig
from model_loader import load_model
import torch

def run_inference(prompt, model_name: str, generation_params: dict = None, torch_dtype: str = "float16"):
    try:
        model, tokenizer = load_model(model_name, torch_dtype)
    except Exception as e:
        return {"error": f"Failed to load model/tokenizer: {type(e).__name__} - {str(e)}"}

    try:
        device = model.device
        if isinstance(prompt, str):
            prompt = [prompt]

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    except Exception as e:
        return {"error": f"Tokenization error: {type(e).__name__} - {str(e)}"}

    try:
        if not generation_params:
            generation_config = GenerationConfig(do_sample=False)
        else:
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
            )

        decoded = [tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
        return {"generated_texts": decoded}

    except Exception as e:
        return {"error": f"Generation error: {type(e).__name__} - {str(e)}"}
