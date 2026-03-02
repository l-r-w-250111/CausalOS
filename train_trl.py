import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with Hugging Face TRL and PEFT.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model directory.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--lora_name", type=str, required=True, help="Name for the output LoRA adapter directory.")
    parser.add_argument("--training_type", type=str, required=True, choices=['cpt', 'sft'], help="Training type: Continued Pre-Training (cpt) or Supervised Fine-Tuning (sft).")
    # Note: Stacked LoRA is complex without a library like Unsloth managing it.
    # We will omit the --cpt_adapter_path for this initial TRL script for simplicity.
    args = parser.parse_args()

    output_dir = f"./lora_models/{args.lora_name}"

    # Quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False # Required for gradient checkpointing

    # PEFT and LoRA setup
    model = prepare_model_for_kbit_training(model)

    # Define target modules based on training type
    sft_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if args.training_type == 'cpt':
        target_modules = sft_target_modules + ["embed_tokens", "lm_head"]
        print(f"CPT training detected. Targeting modules: {target_modules}")
    else:
        target_modules = sft_target_modules
        print(f"SFT training detected. Targeting modules: {target_modules}")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Load dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # Determine dataset text field based on training type
    if args.training_type == 'cpt':
        dataset_text_field = "text"
    else: # For SFT
        if 'instruction' in dataset.column_names:
            dataset_text_field = "instruction"
        elif 'messages' in dataset.column_names:
            # TRL's SFTTrainer can handle chat formats directly
            dataset_text_field = "messages"
        else:
            dataset_text_field = "text" # Fallback
    print(f"Using dataset_text_field: '{dataset_text_field}'")

    # Trainer setup
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=dataset_text_field if dataset_text_field != "messages" else None, # Pass None if using chat format
        max_seq_length=4096,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    # Handle chatML format if necessary (common in SFT datasets)
    if dataset_text_field == "messages" and hasattr(tokenizer, "apply_chat_template"):
        print("Applying ChatML template for 'messages' field.")
    else:
        print("Not applying ChatML template.")


    # Start training
    print("--- Starting TRL Training ---")
    trainer.train()
    print("--- Training Finished ---")

    # Save the fine-tuned LoRA model
    print(f"Saving LoRA model to {output_dir}")
    trainer.save_model(output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    main()
