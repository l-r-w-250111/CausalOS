import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re

class CausalNodeExtractor:
    def __init__(self, model=None, tokenizer=None, model_id="Qwen/Qwen2.5-7B"):
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, device_map="auto"
            )

    def extract_nodes(self, text):
        prompt = f"""Extract main entities (nouns) and actions (verbs) from the following text as causal nodes.
Return only the result as a JSON list of strings.
Example: "A man walks on a street." -> ["man", "walk", "street"]

Text: {text}
JSON:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        # Parse JSON from response
        try:
            match = re.search(r'(\[.*\])', response, re.DOTALL)
            if match:
                raw_nodes = json.loads(match.group(1))
                # Ensure it is a list of strings
                nodes = []
                for n in raw_nodes:
                    if isinstance(n, str):
                        nodes.append(n)
                    elif isinstance(n, dict):
                        # Extract string values from dict if LLM misbehaves
                        nodes.extend([str(v) for v in n.values() if isinstance(v, str)])
                return nodes
            else:
                return []
        except:
            # Fallback: find all words
            return re.findall(r'\w+', response)

if __name__ == "__main__":
    # For testing, we might not want to load the whole model here if it's already loaded elsewhere.
    # But this is a standalone module.
    print("CausalNodeExtractor loaded.")
