import argparse
import os
import shutil
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from llmcompressor.entrypoints import oneshot

def main():
    """
    Merges LoRA adapters and quantizes a model to AWQ format using llm-compressor.
    """
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters and quantize a model to AWQ format using llm-compressor."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the base Hugging Face model directory."
    )
    parser.add_argument(
        "--lora_names", type=str, nargs='*',
        help="Optional list of LoRA adapter names under './lora_models'."
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Directory path to save the quantized AWQ model."
    )
    args = parser.parse_args()

    temp_calib_dir = "./temp_calibration_data"

    try:
        # Step 1: Load model and tokenizer
        print(f"Loading base model from: {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype="auto", # torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        print("Base model and tokenizer loaded.")

        # Step 2: Merge LoRA adapters if provided
        if args.lora_names:
            print(f"Merging {len(args.lora_names)} LoRA adapter(s)...")
            for lora_name in args.lora_names:
                lora_path = os.path.join("./lora_models", lora_name)
                if not os.path.isdir(lora_path):
                    raise FileNotFoundError(f"LoRA adapter not found: {lora_path}")
                print(f"Applying LoRA: {lora_name}")
                model = PeftModel.from_pretrained(model, lora_path)
            print("Merging and unloading adapters...")
            model = model.merge_and_unload()
            print("LoRA adapters merged.")
        else:
            print("No LoRA adapters specified.")

        # Step 3: Create a temporary directory and a dummy calibration file inside it
        print(f"Creating temporary calibration directory at: {temp_calib_dir}")
        os.makedirs(temp_calib_dir, exist_ok=True)
        dummy_data_file_path = os.path.join(temp_calib_dir, "calibration.json")
        dummy_calibration_data = {"text": "This is a dummy sentence for calibration."}
        with open(dummy_data_file_path, "w") as f:
            json.dump(dummy_calibration_data, f)

        # Step 4: Define AWQ recipe and run quantization
        print("Starting AWQ quantization with llm-compressor...")
        recipe = """
        modifiers:
            - type: AWQModifier
              params:
                four_bit_quant_type: "asym"
                quant_config:
                  q_group_size: 128
                  w_bit: 4
                  zero_point: true
                apply_transformers_language_modeling_hf_general: true
        """
        
        oneshot(
            model=model,
            tokenizer=tokenizer,
            dataset="json",
            dataset_path=temp_calib_dir,  # Pass the directory path
            recipe=recipe,
            output_dir=args.output_path,
            num_calibration_samples=1,
            max_seq_length=512,
            trust_remote_code_model=True
        )

        print(f"Successfully quantized and saved AWQ model at: {args.output_path}")

    finally:
        # Step 5: Clean up the temporary calibration directory
        print("Cleaning up temporary files...")
        if os.path.exists(temp_calib_dir):
            shutil.rmtree(temp_calib_dir)
            print(f"Removed temporary directory: {temp_calib_dir}")

if __name__ == "__main__":
    main()
