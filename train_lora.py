import argparse
from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import os

def alpaca_prompt(example):
    """Formats an example from the Alpaca dataset into a prompt."""
    return [
        f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    ]

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with Unsloth, with support for stacking LoRA adapters.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model directory.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--lora_name", type=str, required=True, help="Name for the output LoRA adapter directory.")
    parser.add_argument("--training_type", type=str, required=True, choices=['cpt', 'sft'], help="Training type: Continued Pre-Training (cpt) or Supervised Fine-Tuning (sft).")
    parser.add_argument("--cpt_adapter_path", type=str, default=None, help="Path to a pre-trained CPT LoRA adapter to stack SFT on top of.")
    args = parser.parse_args()

    output_dir = f"./lora_models/{args.lora_name}"
    is_stacking = args.training_type == 'sft' and args.cpt_adapter_path is not None

    # Determine the dtype based on hardware support
    # Use bfloat16 if supported, otherwise default to float16 (by passing None)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else None

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=4096,
        dtype=dtype,
        load_in_4bit=True,
    )

    if is_stacking:
        print(f"Performing stacked SFT. Base CPT adapter: {args.cpt_adapter_path}")
        # 1. Load the CPT adapter
        model.load_lora(args.cpt_adapter_path, adapter_name="cpt_adapter")
        print("Loaded CPT adapter.")
        
        # 2. Add a new SFT adapter
        model.add_lora(
            "sft_adapter",
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
        )
        print("Added new SFT adapter.")
        
        # 3. Set the SFT adapter as the training target
        model.set_adapter("sft_adapter")
        print("Set SFT adapter as trainable.")

    else:
        print("Performing standard LoRA training.")
        
        # Define target modules based on training type
        sft_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if args.training_type == 'cpt':
            target_modules = sft_target_modules + ["embed_tokens", "lm_head"]
            print(f"CPT training detected. Targeting modules: {target_modules}")
        else:
            target_modules = sft_target_modules
            print(f"SFT training detected. Targeting modules: {target_modules}")

        # Configure a single LoRA for CPT or non-stacked SFT
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=target_modules,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

    # Load dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    trainer_args = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": dataset,
        "max_seq_length": 4096,
        "dataset_num_proc": 2,
        "packing": False,
        "args": TrainingArguments(
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
    }

    if args.training_type == 'cpt':
        trainer_args["dataset_text_field"] = "text"
        print("Using 'dataset_text_field' for CPT.")
    elif 'messages' in dataset.column_names:
        # Dynamically create a 'text' column from 'messages' for conversational fine-tuning
        def format_chat_template(example):
            example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False)
            return example
        dataset = dataset.map(format_chat_template)
        trainer_args["dataset_text_field"] = "text"
        print("Using 'apply_chat_template' for Harmony/conversational format.")
    elif 'instruction' in dataset.column_names:
        trainer_args["formatting_func"] = alpaca_prompt
        print("Using 'formatting_func' for Alpaca format.")
    else:
        # Default fallback if no other format is detected
        trainer_args["dataset_text_field"] = "text"
        print("Warning: No specific format detected. Defaulting to 'dataset_text_field'.")

    trainer = SFTTrainer(**trainer_args)

    # Start training
    print("--- Starting Training ---")
    trainer.train()
    print("--- Training Finished ---")

    # Save the fine-tuned LoRA model
    if is_stacking:
        print(f"Saving stacked SFT LoRA adapter to {output_dir}")
        model.save_lora(output_dir, adapter_name="sft_adapter")
    else:
        print(f"Saving LoRA model to {output_dir}")
        model.save_pretrained(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    main()
