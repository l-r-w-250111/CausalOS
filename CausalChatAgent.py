print("[Agent] Loading modules...", flush=True)
import torch
import re
import os
print("[Agent] Importing CausalOS components...", flush=True)
from CausalOS_v4 import device

# ==========================================================
# 9) 対話エージェント (CausalChatAgent)
# ==========================================================
class CausalChatAgent:
    def __init__(self, osys):
        self.osys = osys
        self.history = []
        self.system_prompt = """You are an intelligent assistant capable of causal reasoning.
When a user asks a "What if" question or a counterfactual scenario, you must use the CausalOS physics engine to evaluate the outcome.

To use the tool, output a command in this format:
<tool>solve_counterfactual(factual="FACTUAL_SCENARIO", cf="COUNTERFACTUAL_SCENARIO")</tool>

You can also ask for causal inspiration or inventions:
<tool>get_inspiration()</tool>
<tool>get_invention(context="CONTEXT_FOR_INVENTION")</tool>
"""
        # 初期システムプロンプトを設定
        self.history.append({"role": "system", "content": self.system_prompt})

    def chat(self, user_input):
        print(f"\nUser: {user_input}")
        self.history.append({"role": "user", "content": user_input})

        # 1. LLMによる思考とツール生成
        try:
            response = self._generate_response()
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error"

        # 2. ツール使用の検出
        tool_match = re.search(r"<tool>(.*?)\((.*?)\)</tool>", response, re.DOTALL)

        if tool_match:
            print("[Agent] Tool usage detected.")
            cmd = tool_match.group(1)
            args_str = tool_match.group(2)

            tool_output = ""
            if "solve_counterfactual" in cmd:
                # Parse factual and cf from args_str
                f_match = re.search(r'factual="(.*?)"', args_str)
                cf_match = re.search(r'cf="(.*?)"', args_str)
                if f_match and cf_match:
                    factual = f_match.group(1)
                    cf = cf_match.group(1)
                    options = {"A": "Impossible", "B": "Significant Change", "C": "Nothing"}
                    try:
                        res = self.osys.solve_counterfactual(factual, cf, options)
                        tool_output = f"[Tool Output] Counterfactual solved. Selected outcome: {options[res]}"
                    except Exception as e:
                        tool_output = f"[Tool Output] Error: {e}"

            elif "get_inspiration" in cmd:
                try:
                    activations = self.osys.extrapolate_causal_consequences()
                    inspired = [a[0] for a in activations[:3]]
                    tool_output = f"[Tool Output] Causal extrapolation suggests focus on: {', '.join(inspired)}"
                except Exception as e:
                    tool_output = f"[Tool Output] Error: {e}"

            elif "get_invention" in cmd:
                ctx_match = re.search(r'context="(.*?)"', args_str)
                context = ctx_match.group(1) if ctx_match else user_input
                try:
                    spark = self.osys.generate_inventive_spark(context)
                    tool_output = f"[Tool Output] Invention generated.\n{spark}"
                except Exception as e:
                    tool_output = f"[Tool Output] Error: {e}"

            print(f"[Agent] Tool Output: {tool_output.strip()}")
            self.history.append({"role": "assistant", "content": response})
            self.history.append({"role": "tool", "content": tool_output})

            # 3. 最終回答の生成
            final_response = self._generate_response()
            print(f"Assistant: {final_response}")
            self.history.append({"role": "assistant", "content": final_response})
            return final_response

        else:
            # ツール不使用
            print(f"Assistant: {response}")
            self.history.append({"role": "assistant", "content": response})
            return response

    def _generate_response(self):
        prompt = ""
        for msg in self.history:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"{content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
            elif role == "tool":
                prompt += f"System: {content}\n"

        prompt += "Assistant: "

        inputs = self.osys.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.osys.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.osys.tokenizer.eos_token_id
            )
        return self.osys.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

if __name__ == "__main__":
    print("[Agent] Starting main block...", flush=True)
    from CausalOS_v4 import UnifiedCausalOSV4

    print("--- Starting CausalChatAgent (CausalOS v4) ---", flush=True)
    # Initialize with a smaller model for interactive demo
    osys = UnifiedCausalOSV4(model_id="Qwen/Qwen2.5-7B-Instruct")
    agent = CausalChatAgent(osys)

    print("\nReady! Type your message (or 'exit' to quit).", flush=True)
    while True:
        try:
            user_msg = input("\nUser: ")
            if user_msg.lower() in ["exit", "quit", "q"]:
                break

            response = agent.chat(user_msg)
            # Response is already printed inside chat()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
