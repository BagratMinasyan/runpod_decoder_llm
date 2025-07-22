import runpod
from run_inference import run_inference

def handler(event):
    print("🚀 Handler started - received event")
    print(f"📝 Event keys: {list(event.keys())}")
    
    try:
        print("📋 Extracting input parameters...")
        input = event["input"]
        print(f"📋 Input keys: {list(input.keys())}")
        
        prompt = input.get("prompt")
        model_name = input.get("model_name")
        torch_dtype = input.get("torch_dtype", "float16")
        generation_params = input.get("generation_config", {})
        
        print(f"📝 Prompt: {prompt[:100] if prompt else None}{'...' if prompt and len(prompt) > 100 else ''}")
        print(f"🤖 Model name: {model_name}")
        print(f"🔢 Torch dtype: {torch_dtype}")
        print(f"⚙️ Generation params: {generation_params}")

        if not prompt or not model_name:
            print("❌ Missing required parameters")
            return {"error": "Missing 'prompt' or 'model_name'"}

        print("🔄 Calling run_inference...")
        result = run_inference(
            prompt=prompt,
            model_name=model_name,
            generation_params=generation_params,
            torch_dtype=torch_dtype
        )
        
        print("✅ Handler completed successfully")
        return result

    except Exception as e:
        print(f"💥 Exception in handler: {type(e).__name__} - {str(e)}")
        return {"error": f"Unhandled exception in handler: {type(e).__name__} - {str(e)}"}

if __name__ == "__main__":
    print("🏁 Starting RunPod serverless...")
    runpod.serverless.start({"handler": handler})