import runpod
from inference import run_inference

def handler(event):
    input = event["input"]
    prompt = input.get("prompt")
    model_name = input.get("model_name")
    torch_dtype = input.get("torch_dtype", "float16")
    generation_params = input.get("generation_config", {})

    if not prompt or not model_name:
        return {"error": "Missing 'prompt' or 'model_name'"}

    result = run_inference(
        prompt=prompt,
        model_name=model_name,
        generation_params=generation_params,
        torch_dtype=torch_dtype
    )
    return result

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
