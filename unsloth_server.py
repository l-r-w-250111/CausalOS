from unsloth import FastLanguageModel
import os
import sys
import torch
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from starlette.responses import StreamingResponse
from transformers import TextIteratorStreamer
from threading import Thread


# Suppress Unsloth's welcome message on startup
os.environ["UNSLOTH_SUPPRESS_STDOUT"] = "true"

# Global variables to hold the model and tokenizer
model = None
tokenizer = None

# --- FastAPI App Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This block runs on startup
    global model, tokenizer
    print("Starting Unsloth server lifespan...", file=sys.stderr)

    # --- Environment Variable Parsing ---
    base_model_path = os.getenv("BASE_MODEL_PATH")
    lora_model_path = os.getenv("LORA_MODEL_PATH", None)

    if not base_model_path:
        print("FATAL: BASE_MODEL_PATH environment variable is not set.", file=sys.stderr)
        # The server will start, but endpoints will fail with 503.
        # This allows the container to run and logs to be inspected.
        yield
        return # Exit the lifespan manager cleanly

    print(f"Loading base model: {base_model_path}", file=sys.stderr)
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        if lora_model_path and os.path.isdir(lora_model_path):
            print(f"Loading and applying LoRA adapter from: {lora_model_path}", file=sys.stderr)
            model.load_adapter(lora_model_path)
            print("LoRA adapter applied successfully.", file=sys.stderr)

        model.eval()
        FastLanguageModel.for_inference(model)
        print("Model loaded successfully.", file=sys.stderr)

    except Exception as e:
        print(f"FATAL: Failed to load model. Error: {e}", file=sys.stderr)
        # Server will start but endpoints will fail.

    yield
    # This block runs on shutdown
    print("Shutting down Unsloth server...", file=sys.stderr)
    model = None
    tokenizer = None


app = FastAPI(lifespan=lifespan)

# --- Pydantic Models for Request/Response ---
from typing import List, Dict

class GenerateRequest(BaseModel):
    conversation: List[Dict[str, str]]

class GenerateNonStreamingRequest(BaseModel):
    prompt: str

class GenerateNonStreamingResponse(BaseModel):
    response: str

# --- API Endpoints ---
@app.get("/health")
async def health_check():
    """Health check endpoint to verify the server is running and the model is loaded."""
    if model is not None and tokenizer is not None:
        return {"status": "ok", "message": "Server and model are ready."}
    else:
        if not os.getenv("BASE_MODEL_PATH"):
            raise HTTPException(status_code=503, detail="Server is running, but model is not loaded: BASE_MODEL_PATH env var was not set.")
        else:
            raise HTTPException(status_code=503, detail="Server is running, but model failed to load. Check server logs.")

@app.post("/generate")
async def generate_text_stream(request: GenerateRequest):
    """Generates text based on a prompt and streams the response."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot process request.")

    try:
        final_prompt = tokenizer.apply_chat_template(request.conversation, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(final_prompt, return_tensors="pt", padding=True).input_ids.to("cuda") #2512211943ADD padding=True
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=2048, # Increased max_new_tokens
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        async def response_generator():
            try:
                for new_text in streamer:
                    yield new_text
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"Caught CUDA out of memory error: {e}", file=sys.stderr)
                    # We can't yield an error message here as the headers are already sent.
                    # The client will experience a broken connection, which we will handle.
                else:
                    raise e # Re-raise other runtime errors

        return StreamingResponse(response_generator(), media_type="text/plain")

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"Error during streaming generation setup: {e}", file=sys.stderr)
            raise HTTPException(status_code=500, detail="CUDA out of memory")
        else:
            print(f"Unhandled RuntimeError during streaming generation: {e}", file=sys.stderr)
            raise HTTPException(status_code=500, detail=f"An unexpected runtime error occurred: {str(e)}")
    except Exception as e:
        print(f"Error during streaming generation: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"An error occurred during text generation: {str(e)}")

@app.post("/generate_non_streaming", response_model=GenerateNonStreamingResponse)
async def generate_text_non_streaming(request: GenerateNonStreamingRequest):
    """Generates a short, non-streamed response, intended for routing or quick checks."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot process request.")

    try:
        # For routing, apply the chat template to ensure the model understands the instruction format.
        conversation = [{"role": "user", "content": request.prompt}]
        final_prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(final_prompt, return_tensors="pt", padding=True).input_ids.to("cuda") #2512211943ADD padding=True

        with torch.no_grad():
            # Generate a short response suitable for a routing decision
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=10, # Keep it short and fast for routing
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][input_ids.shape[1]:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return GenerateNonStreamingResponse(response=response_text)

    except Exception as e:
        print(f"Error during non-streaming generation: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"An error occurred during text generation: {str(e)}")

# This part is for direct execution, but we'll use uvicorn via docker compose exec
if __name__ == "__main__":
    import uvicorn
    # This is for local debugging only. The app will be started by `uvicorn unsloth_server:app ...`
    # Environment variables (BASE_MODEL_PATH) must be set before running.
    uvicorn.run(app, host="0.0.0.0", port=8003)
