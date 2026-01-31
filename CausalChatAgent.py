import torch
import re
import os
from CausalOS_v1 import device

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

Example:
User: "If a plane landed in the forest instead of a bird, what happens?"
Assistant: <tool>solve_counterfactual(factual="A bird lands in a forest", cf="A plane lands in a forest")</tool>
"""
        # 初期システムプロンプトを設定
        self.history.append({"role": "system", "content": self.system_prompt})

    def chat(self, user_input):
        print(f"\nUser: {user_input}")
        self.history.append({"role": "user", "content": user_input})
        
        # 1. LLMによる思考とツール生成
        try:
            response = self._generate_response()
        except AttributeError as e:
            # エラーハンドリング: モデルへのアクセスを試行錯誤時に失敗しやすいため
            print(f"Error generating response: {e}")
            return "Error"

        # 2. ツール使用の検出
        tool_match = re.search(r"<tool>solve_counterfactual\(factual=\"(.*?)\", cf=\"(.*?)\"\)</tool>", response, re.DOTALL)
        
        if tool_match:
            print("[Agent] Tool usage detected.")
            factual = tool_match.group(1)
            cf = tool_match.group(2)
            
            # CausalOSの実行
            options = {
                "A": "Nothing special happens.",
                "B": "A serious accident or damage occurs. (High Severity)",
                "C": "It is impossible.",
                "D": "The situation improves."
            }
            
            # 実行
            try:
                result_key = self.osys.solve_counterfactual(factual, cf, options)
                
                # 詳細情報の取得（UnifiedCausalOSは文字列キーのみ返す）
                selected_text = options.get(result_key, "Unknown option")
                
                tool_output = f"""
[Tool Output]
Selected Option: {selected_text} (Key: {result_key})
Physics Reasoning: (Internal simulation completed. Check logs for details.)
"""
            except Exception as e:
                tool_output = f"[Tool Output] Error executing CausalOS: {e}"

            print(f"[Agent] Tool Output: {tool_output.strip()}")
            self.history.append({"role": "assistant", "content": response}) # 思考過程
            self.history.append({"role": "tool", "content": tool_output})
            
            # 3. 最終回答の生成
            final_response = self._generate_response()
            print(f"Assistant: {final_response}")
            self.history.append({"role": "assistant", "content": final_response})
            return final_response
            
        else:
            # ツール不使用（通常の会話）
            print(f"Assistant: {response}")
            self.history.append({"role": "assistant", "content": response})
            return response

    def _generate_response(self):
        # 履歴をプロンプトに変換
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
            # self.osys.model にアクセス (Step 407 修正反映)
            outputs = self.osys.model.generate(
                **inputs, 
                max_new_tokens=150, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=self.osys.tokenizer.eos_token_id
            )
        return self.osys.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
