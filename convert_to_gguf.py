from unsloth import FastLanguageModel
import argparse
import torch
import os
import shutil
import subprocess

# Initialize Unsloth
FastLanguageModel.disable_fast_init()
FastLanguageModel.for_inference()

def run_command(command, description):
    """Executes a shell command and raises an exception on failure."""
    print(f"Executing: {description}")
    print(f"Command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"Error executing {description}:")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"{description} failed.")
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}") # Print stderr even on success for warnings
    print(f"{description} completed successfully.")
    return result

def main():
    parser = argparse.ArgumentParser(description="Convert a Hugging Face model to GGUF format using llama.cpp.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model directory.")
    parser.add_argument("--lora_names", type=str, nargs='*', help="Optional: List of LoRA adapter names under lora_models/.")
    parser.add_argument("--quantization_method", type=str, required=True, help="Quantization method (e.g., q4_k_m).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the final GGUF file.")
    parser.add_argument("--is_4bit", action="store_true", help="Flag if the base model is 4-bit. This is used to determine the loading method.")
    args = parser.parse_args()

    # Define paths
    temp_fp16_path = "./temp_fp16_model"
    # Ensure the parent directory for the output file exists
    output_parent_dir = os.path.dirname(args.output_path)
    if output_parent_dir:
        os.makedirs(output_parent_dir, exist_ok=True)

    # Create an intermediate path in the same directory as the final output
    intermediate_gguf_path = os.path.join(output_parent_dir, "temp_model_f16.gguf")

    try:
        # --- 1. Load Model and Merge LoRA using Unsloth ---
        print("Loading base model and applying LoRA adapter (if specified)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            load_in_4bit=args.is_4bit,
            # Using bfloat16 is fine for loading, we'll save in float16
            dtype=torch.bfloat16 if not args.is_4bit else None,
        )

        if args.lora_names:
            print(f"Applying {len(args.lora_names)} LoRA adapter(s)...")
            for lora_name in args.lora_names:
                lora_path = os.path.join("./lora_models", lora_name)
                if not os.path.isdir(lora_path):
                    raise FileNotFoundError(f"LoRA adapter directory not found: {lora_path}")
                print(f"Loading adapter: {lora_name}")
                model.load_adapter(lora_path)

            print("All LoRA adapters loaded. Merging...")
            # Unsloth's merge_and_unload will merge all loaded adapters.
            model.merge_and_unload()
            print("LoRA adapters successfully merged.")


        # --- 2. Save Merged Model to FP16 Hugging Face Format ---
        print(f"Saving merged model to temporary FP16 directory: {temp_fp16_path}")
        if os.path.exists(temp_fp16_path):
            shutil.rmtree(temp_fp16_path)

        model.save_pretrained(temp_fp16_path, save_dtype=torch.float16)
        tokenizer.save_pretrained(temp_fp16_path)
        print("Model saved in FP16 format successfully.")

        # --- 3. Convert FP16 Model to FP16 GGUF using llama.cpp ---
        convert_command = [
            "python", "/opt/llama.cpp/convert.py", temp_fp16_path,
            "--outfile", intermediate_gguf_path,
            "--outtype", "f16" # Create an intermediate f16 GGUF
        ]
        run_command(convert_command, "FP16 Hugging Face to FP16 GGUF conversion")

        # --- 4. Quantize FP16 GGUF to Target Format using llama.cpp ---
        quantize_command = [
            "/opt/llama.cpp/build/bin/quantize",
            intermediate_gguf_path,
            args.output_path,
            args.quantization_method
        ]
        run_command(quantize_command, "GGUF Quantization")

        print(f"Successfully created quantized GGUF model at: {args.output_path}")

    finally:
        # --- 5. Cleanup ---
        print("Cleaning up temporary files...")
        if os.path.exists(temp_fp16_path):
            shutil.rmtree(temp_fp16_path)
            print(f"Removed temporary directory: {temp_fp16_path}")
        if os.path.exists(intermediate_gguf_path):
            os.remove(intermediate_gguf_path)
            print(f"Removed intermediate GGUF file: {intermediate_gguf_path}")

if __name__ == "__main__":
    main()
