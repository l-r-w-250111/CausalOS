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
        """Deprecated in favor of extract_label_graph, but kept for compatibility"""
        g = self.extract_label_graph(text)
        return [l["name"] for l in g.get("labels", [])]

    def extract_label_graph(self, text):
        prompt = f"""Text: {text}
Extract entities/actions as JSON.
Example: {{
  "labels": [{{"name": "man", "nodes": 2}}, {{"name": "walk", "nodes": 2}}],
  "edges": [{{"src": "man", "dst": "walk"}}]
}}
JSON:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=150, 
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        try:
            match = re.search(r'(\{.*\})', response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            else:
                return {"labels": [], "edges": []}
        except:
            return {"labels": [], "edges": []}

if __name__ == "__main__":
    print("CausalNodeExtractor loaded.")
