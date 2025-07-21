import runpod
from run_inference import run_inference

def handler(event):
    try:
        data = event.get("input", {})
        prompt = data.get("prompt")
        torch_dtype = data.get("torch_dtype", "float16")
        generation_params = data.get("generation_config", {})

        if not prompt:
            return {"error": "Missing 'prompt' in input"}

        return run_inference(
            prompt=prompt,
            generation_params=generation_params,
            torch_dtype=torch_dtype
        )

    except Exception as e:
        return {
            "error": f"Unhandled exception: {type(e).__name__} – {e}"
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
