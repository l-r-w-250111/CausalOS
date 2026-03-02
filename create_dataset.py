import argparse
import json
import os
from llama_index.core import SimpleDirectoryReader

def load_chat_history(chat_history_path):
    """Loads chat history from the structured JSON file."""
    if not os.path.exists(chat_history_path):
        print(f"Warning: Chat history file not found at {chat_history_path}")
        return []
    with open(chat_history_path, 'r', encoding='utf-8') as f:
        try:
            all_sessions = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {chat_history_path}")
            return []

    combined_history = []
    for session_id, session_data in all_sessions.items():
        if isinstance(session_data, dict) and 'messages' in session_data:
            combined_history.extend(session_data['messages'])
    return combined_history

def convert_to_alpaca(chat_history):
    """Converts paired user-assistant messages to Alpaca format."""
    alpaca_data = []
    for i in range(0, len(chat_history) - 1, 2):
        if chat_history[i]['role'] == 'user' and chat_history[i+1]['role'] == 'assistant':
            alpaca_data.append({
                "instruction": chat_history[i]['content'],
                "input": "",
                "output": chat_history[i+1]['content']
            })
    return alpaca_data

def convert_to_harmony(chat_history):
    """Converts paired user-assistant messages to Harmony chat format."""
    harmony_data = []
    for i in range(0, len(chat_history) - 1, 2):
        if chat_history[i]['role'] == 'user' and chat_history[i+1]['role'] == 'assistant':
            harmony_data.append({"messages": [chat_history[i], chat_history[i+1]]})
    return harmony_data

def create_dataset(output_path, format_type, chat_history_path=None, source_dir=None):
    """
    Creates a dataset from various sources based on the specified format.
    - For Alpaca/Harmony, uses chat history.
    - For CPT, concatenates text from all files in a source directory.
    """
    print(f"Creating dataset in '{format_type}' format.")
    final_dataset = []
    source_dirs = source_dir if isinstance(source_dir, list) else [source_dir]


    if format_type in ["alpaca", "harmony"]:
        chat_history = []
        if chat_history_path:
            print(f"Loading chat history from: {chat_history_path}")
            chat_history = load_chat_history(chat_history_path)

        # SFT format can also come from files, let's process source_dir if provided
        if source_dirs and all(d for d in source_dirs):
            for directory in source_dirs:
                if os.path.isdir(directory):
                    print(f"Loading SFT data from files in: {directory}")
                    try:
                        documents = SimpleDirectoryReader(directory, recursive=True).load_data()
                    except Exception as e:
                        print(f"Warning: Could not process files in {directory}: {e}")


        if not chat_history:
            print("Warning: No chat history provided or found. The dataset will be based on other sources or empty.")

        if format_type == "alpaca":
            final_dataset.extend(convert_to_alpaca(chat_history))
        elif format_type == "harmony":
            final_dataset.extend(convert_to_harmony(chat_history))

    elif format_type == "cpt":
        if not source_dirs or not all(d for d in source_dirs):
            raise ValueError("Source directory is required for CPT format.")

        all_text_content = []
        for directory in source_dirs:
            if not os.path.isdir(directory):
                print(f"Warning: Source directory not found, skipping: {directory}")
                continue

            print(f"Extracting text for CPT from directory: {directory}")
            try:
                documents = SimpleDirectoryReader(directory, recursive=True).load_data()
                content = "\n\n".join([doc.get_content() for doc in documents])
                if content:
                    all_text_content.append(content)
            except Exception as e:
                print(f"Error processing source directory {directory}: {e}")

        if all_text_content:
            final_dataset = [{"text": "\n\n".join(all_text_content)}]
        else:
            print("Warning: No text content could be extracted from the source directories.")

    # Save the final dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
        f.write('\n')

    print(f"Dataset with {len(final_dataset)} entries saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a fine-tuning dataset from various sources.")
    parser.add_argument("--output_path", required=True, help="Path to save the generated dataset.")
    parser.add_argument("--format_type", required=True, choices=["alpaca", "harmony", "cpt"], help="The format of the dataset.")
    parser.add_argument("--chat_history_path", type=str, default=None, help="Path to the chat history file (required for Alpaca/Harmony).")
    parser.add_argument("--source_dir", type=str, nargs='+', default=None, help="Source directory with documents for CPT.")

    args = parser.parse_args()

    create_dataset(
        output_path=args.output_path,
        format_type=args.format_type,
        chat_history_path=args.chat_history_path,
        source_dir=args.source_dir
    )
