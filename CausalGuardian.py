import torch
import torch.nn.functional as F
import re

# UnifiedCausalOS_v3 (Physics Latent Space) をインポート
from CausalOS_v3 import UnifiedCausalOS_v3, device

class CausalGuardian:
    def __init__(self, causal_os):
        """
        Args:
            causal_os: UnifiedCausalOS_v3 インスタンス
        """
        print("[Guardian] Initializing Causal Guardian V3 (Latent Physics Core)...")
        self.osys = causal_os
        self.tokenizer = self.osys.tokenizer
        self.model = self.osys.model
        
        self.phi_history = []
        self.noise_scale = 0.02
    
    # ... (Phi, CII, Mass calculations remain the same) ...
    def calculate_phi(self, logits):
        probs = F.softmax(logits, dim=-1)
        top_v, _ = torch.topk(probs, 50)
        phi = 1.0 / (torch.var(top_v).item() + 1e-6)
        return phi

    def calculate_cii(self):
        if len(self.phi_history) < 3:
            return 0.0
        d2_phi = (self.phi_history[-1] - 2 * self.phi_history[-2] + self.phi_history[-3])
        return d2_phi ** 2

    def calculate_mass(self, input_ids, seed=None):
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            original_logits = outputs.logits[0, -1, :]
            original_probs = F.softmax(original_logits, dim=-1)

        embeddings = self.model.get_input_embeddings()
        inputs_embeds = embeddings(input_ids)
        
        if seed is not None:
            torch.manual_seed(seed)
            
        noise = torch.randn_like(inputs_embeds) * self.noise_scale
        perturbed_embeds = inputs_embeds + noise
        
        with torch.no_grad():
            p_outputs = self.model(inputs_embeds=perturbed_embeds)
            p_logits = p_outputs.logits[0, -1, :]
            p_log_probs = F.log_softmax(p_logits, dim=-1)
            
        p_log_probs = torch.clamp(p_log_probs, min=-100.0)
        kl_div = F.kl_div(p_log_probs, original_probs, reduction='sum').item()
        kl_div = max(0.0, kl_div)
        
        mass = 1.0 / (kl_div + 1e-4)
        return mass, kl_div

    def generate_with_monitoring(self, prompt, max_tokens=100, cii_threshold=2000.0, mass_threshold=30.0, seed=42):
        print(f"\n[Guardian] Generating with Latent Physics Monitoring")
        print(f"Thresholds: CII > {cii_threshold} OR Mass < {mass_threshold}")
        print(f"{'-'*110}")
        print(f"{'Step':<4} | {'Phi':<8} | {'CII':<10} | {'Mass':<8} | {'KL':<8} | {'Token':<15} | {'Status'}")
        print(f"{'-'*110}")
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated_ids = input_ids.clone()
        
        self.phi_history = []
        intervention_count = 0
        max_interventions = 3
        self.last_intervention_step = -1
        self.last_concerns = set()
        
        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(generated_ids)
                logits = outputs.logits[0, -1, :]
                
                phi = self.calculate_phi(logits)
                self.phi_history.append(phi)
                cii = self.calculate_cii()
                
                mass, kl = self.calculate_mass(generated_ids, seed=seed+step)
                
                mass, kl = self.calculate_mass(generated_ids, seed=seed+step)
                
                # Sampling instead of Greedy to avoid loops
                temperature = 0.7
                if temperature > 0:
                     probs = F.softmax(logits / temperature, dim=-1)
                     next_token_id = torch.multinomial(probs, num_samples=1).item()
                else:
                     next_token_id = torch.argmax(logits).item()

                token_str = self.tokenizer.decode(next_token_id)
                
                status = ""
                
                trigger_intervention = False
                if intervention_count < max_interventions:
                    if mass < mass_threshold:
                        status = "⚠ LOW MASS"
                        trigger_intervention = True
                    elif cii > cii_threshold:
                        status = "⚠ HIGH CII"
                        trigger_intervention = True
                
                print(f"{step:03}  | {phi:<8.2f} | {cii:<10.2f} | {mass:<8.2f} | {kl:<8.4f} | [{token_str:<13}] | {status}")
                
                # Check for analysis trigger (only if not in cooldown)
                # Avoid redundant analysis if 2 alerts already exist
                current_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                num_alerts = current_text.count("[System Alert")
                in_cooldown = (self.last_intervention_step >= 0 and (step - self.last_intervention_step) < 15)
                
                if trigger_intervention and not in_cooldown and num_alerts < 2:
                    analysis = self._run_physics_analysis(current_text)
                    
                    if analysis and analysis["severity"] > 0.4:  # Threshold for intervention
                        print(f"\n{'='*110}")
                        print(f"[Guardian] 🛡 LATENT PHYSICS INTERVENTION TRIGGERED")
                        print(f"  Severity: {analysis['severity']:.2f}")
                        print(f"  Micro-concerns: {', '.join(analysis['concerns'][:2])}")
                        print(f"  Recommendation: {analysis['recommendation']}")
                        print(f"{'='*110}\n")
                        
                        # Check duplicate concerns (Smart Suppression - Normalized)
                        norm_concerns = set([re.sub(r'\(.*?\)', '', c).strip() for c in analysis['concerns']])
                        if norm_concerns and norm_concerns == self.last_concerns:
                             print("[Guardian] ℹ Same concerns as previous, skipping redundant intervention")
                             self.last_intervention_step = step # Reset cooldown but don't inject
                             # Note: We fall through to append the token below
                        else:
                             self.last_concerns = norm_concerns
                             injection = self._create_intervention_prompt(analysis)
                             injection_ids = self.tokenizer.encode(injection, return_tensors="pt").to(device)
                             generated_ids = torch.cat([generated_ids, injection_ids], dim=-1)
                             
                             intervention_count += 1
                             self.last_intervention_step = step
                             self.phi_history = []
                             continue # Skip token append for intervention step
                        
                generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]], device=device)], dim=-1)
                
                if next_token_id == self.tokenizer.eos_token_id:
                    break
        
        print(f"{'-'*110}\n")
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def _run_physics_analysis(self, text):
        """物理的整合性分析を実行"""
        try:
            # Clean text to remove Guardian's own injections (avoid self-analysis loop)
            # Remove content from [System ... to Assistant:
            clean_text = re.sub(r'\[System.*?Assistant:', '', text, flags=re.DOTALL | re.IGNORECASE)
            # Also remove simple [Physics: ...] blocks
            clean_text = re.sub(r'\[Physics.*?Assistant:', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
            
            return self.osys.analyze_physical_consistency(clean_text)
        except Exception as e:
            print(f"[Guardian] Physics analysis error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_intervention_prompt(self, analysis):
        """分析結果（潜在メトリクス）から介入プロンプトを生成"""
        severity = analysis["severity"]
        concerns = analysis["concerns"]
        metrics = analysis.get("latent_metrics", {})
        
        if severity > 0.7:
            prompt_parts = ["\n\n[System Alert: CRITICAL PHYSICAL INCONSISTENCY]"]
            if concerns:
                prompt_parts.append("Latent Space Anomalies Detected:")
                for concern in concerns[:3]:
                    prompt_parts.append(f"  • {concern}")
            
            if metrics:
                prompt_parts.append("\nLatent Metrics:")
                if "max_distance" in metrics and metrics["max_distance"] > 1.0:
                    prompt_parts.append(f"  • Vector Distance: {metrics['max_distance']:.2f} (Extreme divergence)")
                if "max_magnitude" in metrics and metrics["max_magnitude"] > 0.8:
                    prompt_parts.append(f"  • Force Magnitude: High ({metrics['max_magnitude']:.2f})")
            
            prompt_parts.extend([
                "\nGuidance:",
                "1. Re-evaluate the interaction in physical latent space.",
                "2. The structural integrity of the objects (vector stability) is compromised.",
                "3. Conclude with a safety-first outcome considering these force magnitudes.",
                "\nAssistant:"
            ])
            
        elif severity > 0.4:
            prompt_parts = ["\n\n[System Note: Physical Plausibility Check]"]
            if concerns:
                prompt_parts.append(f"Observation: {concerns[0]}")
            prompt_parts.append("\nGuidance: Ensure the outcome respects mass/energy constraints.\nAssistant:")
            
        else:
            return "\n[Physics: Stable]\nAssistant:"
        
        return "\n".join(prompt_parts)

if __name__ == "__main__":
    from CausalOS_v3 import UnifiedCausalOS_v3
    osys = UnifiedCausalOS_v3(n_nodes=10)
    guardian = CausalGuardian(osys)
    
    query = "If a massive Boeing 747 slowly landed on a single oak tree branch, "
    print(guardian.generate_with_monitoring(query, max_tokens=150, mass_threshold=50.0, seed=42))
