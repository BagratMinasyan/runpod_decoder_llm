import torch
from transformers import GenerationConfig
from model_loader import load_model

def run_inference(prompt, generation_params: dict = None, torch_dtype: str = "float16"):
    # (Re)load model with the requested dtype
    model, tokenizer = load_model(torch_dtype)
    device = model.device

    # Support single string or list of strings
    if isinstance(prompt, str):
        prompt = [prompt]

    # Tokenize with padding/truncation
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # Build GenerationConfig:
    #  - if no params given, generate deterministically
    if not generation_params:
        gen_cfg = GenerationConfig(do_sample=False)
    else:
        gen_cfg = GenerationConfig(**generation_params)

    # Generate!
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=gen_cfg,
            return_dict_in_generate=True,
            output_scores=False
        )

    # Decode all sequences
    decoded = [
        tokenizer.decode(seq, skip_special_tokens=True)
        for seq in outputs.sequences
    ]

    return {"generated_texts": decoded}
