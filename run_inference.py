from transformers import GenerationConfig
from model_loader import load_model
import torch

def run_inference(prompt, model_name: str, generation_params: dict = None, torch_dtype: str = "float16"):
    model, tokenizer = load_model(model_name, torch_dtype)
    device = model.device

    if isinstance(prompt, str):
        prompt = [prompt]

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    if not generation_params:
        generation_config = GenerationConfig(do_sample=False)
    else:
        generation_config = GenerationConfig(**generation_params)

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
