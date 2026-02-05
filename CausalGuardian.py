import torch
import torch.nn.functional as F
import re
import numpy as np

# UnifiedCausalOSV4 (Causal Latent Space) をインポート
from CausalOS_v4 import UnifiedCausalOSV4, device

class AdaptiveAnomalyDetector:
    """動的閾値と多次元指標を組み合わせた違和感検出器"""

    def __init__(self, window_size=20, sensitivity=1.5):
        self.window_size = window_size
        self.sensitivity = sensitivity

    def detect_anomaly(self, phi_history, cii_history, current_step):
        """
        多次元的な違和感検出
        - Phi: 状態の不確実性（CSIの代用）
        - CII: 位相加速度
        """
        if len(phi_history) < 5:
            return False, 0.0

        # 動的ベースライン更新 (最近の窓から計算)
        recent_phi = np.array(phi_history[-self.window_size:])
        mu_phi = np.mean(recent_phi)
        sigma_phi = np.std(recent_phi)

        # Zスコア (急激な不確実性の変化)
        z_score = abs((phi_history[-1] - mu_phi) / (sigma_phi + 1e-8))

        # 加速度（CIIの二次微分相当）
        # cii自体が加速度的なので、その変化を見る
        acceleration = 0.0
        if len(cii_history) >= 3:
            # ciiの差分の差分
            acceleration = np.diff(cii_history[-3:], n=2)[-1]

        # Phase space trajectory curvature
        trajectory_curve = 0.0
        if len(phi_history) >= 3 and len(cii_history) >= 3:
            trajectory_curve = self._compute_curvature(
                phi_history[-3:], cii_history[-3:]
            )

        # 統合スコア (正規化しつつ統合)
        # phi_historyの絶対値が大きいので、Zスコアベースを主にする
        anomaly_score = (
            0.5 * min(5.0, z_score) +
            0.2 * min(5.0, abs(acceleration) / 1000.0) +
            0.3 * min(5.0, trajectory_curve)
        )

        return anomaly_score > self.sensitivity, anomaly_score

    def _compute_curvature(self, x, y):
        """位相空間軌道の曲率計算"""
        x = np.array(x)
        y = np.array(y)
        dx = np.diff(x)
        dy = np.diff(y)

        if len(dx) < 2:
            return 0.0

        ddx = np.diff(dx)
        ddy = np.diff(dy)

        if len(ddx) == 0:
            return 0.0

        numerator = abs(dx[-1] * ddy[0] - dy[-1] * ddx[0])
        denominator = (dx[-1]**2 + dy[-1]**2)**1.5
        return numerator / (denominator + 1e-8)

class CausalGuardian:
    """
    Guardian with Dynamic Anomaly Detection using AdaptiveAnomalyDetector.
    """
    def __init__(self, causal_os, detector_sensitivity=2.5):
        """
        Args:
            causal_os: UnifiedCausalOSV4 インスタンス
        """
        print("[Guardian] Initializing Causal Guardian V4 (Dynamic Anomaly Detection)...")
        self.osys = causal_os
        self.tokenizer = self.osys.tokenizer
        self.model = self.osys.model

        self.phi_history = []
        self.cii_history = []
        self.mass_history = []
        self.entropy_history = []
        self.top_k_entropy_history = []
        self.prob_margin_history = []
        self.logit_drift_history = []
        self.last_logits = None

        self.noise_scale = 0.02
        self.last_intervention_step = -1
        self.last_concerns = set()
        self.detector = AdaptiveAnomalyDetector(sensitivity=detector_sensitivity)

    def calculate_phi(self, logits):
        probs = F.softmax(logits.float(), dim=-1)
        top_v, _ = torch.topk(probs, 50)
        phi = 1.0 / (torch.var(top_v).item() + 1e-6)
        return phi

    def calculate_top_k_entropy(self, logits, k=10):
        probs = F.softmax(logits.float(), dim=-1)
        top_probs, _ = torch.topk(probs, k)
        top_probs = top_probs / torch.sum(top_probs) # Re-normalize
        entropy = -torch.sum(top_probs * torch.log(top_probs + 1e-10)).item()
        return entropy

    def calculate_prob_margin(self, logits):
        probs = F.softmax(logits.float(), dim=-1)
        top_probs, _ = torch.topk(probs, 2)
        margin = (top_probs[0] - top_probs[1]).item()
        return margin

    def calculate_logit_drift(self, logits):
        if self.last_logits is None:
            self.last_logits = logits.detach().clone()
            return 0.0
        
        drift = torch.norm(logits - self.last_logits, p=2).item()
        self.last_logits = logits.detach().clone()
        return drift

    def calculate_cii(self):
        if len(self.phi_history) < 3:
            return 0.0
        d2_phi = (self.phi_history[-1] - 2 * self.phi_history[-2] + self.phi_history[-3])
        return d2_phi ** 2

    def calculate_mass(self, input_ids, seed=None):
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            original_logits = outputs.logits[0, -1, :]
            original_probs = F.softmax(original_logits.float(), dim=-1)

        embeddings = self.model.get_input_embeddings()
        inputs_embeds = embeddings(input_ids)

        if seed is not None:
            torch.manual_seed(seed)

        noise = torch.randn_like(inputs_embeds) * self.noise_scale
        perturbed_embeds = inputs_embeds + noise

        with torch.no_grad():
            p_outputs = self.model(inputs_embeds=perturbed_embeds)
            p_logits = p_outputs.logits[0, -1, :]
            p_log_probs = F.log_softmax(p_logits.float(), dim=-1)

        p_log_probs = torch.clamp(p_log_probs, min=-100.0)
        kl_div = F.kl_div(p_log_probs, original_probs, reduction='sum').item()
        kl_div = max(0.0, kl_div)

        mass = 1.0 / (kl_div + 1e-4)
        return mass, kl_div

    def get_dynamic_threshold(self, history, window=10, multiplier=3.0, default=1.0, lower_bound=True):
        if len(history) < window:
            return default
        recent = np.array(history[-window:])
        mean = np.mean(recent)
        std = np.std(recent)
        if lower_bound:
            # For Mass (lower is bad)
            return max(0.01, mean - multiplier * std)
        else:
            # For CII (higher is bad)
            return mean + multiplier * std

    def generate_with_monitoring(self, prompt, max_tokens=100, cii_threshold=2000.0, mass_threshold=30.0, seed=42):
        print(f"\n[Guardian] Generating with Adaptive Anomaly Detection")
        print(f"{'-'*140}")
        print(f"{'Step':<4} | {'Phi':<7} | {'CII':<8} | {'Mass':<7} | {'Ent':<6} | {'T-Ent':<6} | {'Mrg':<6} | {'Drift':<6} | {'A-Score':<8} | {'Token':<15} | {'Status'}")
        print(f"{'-'*140}")

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated_ids = input_ids.clone()

        self.phi_history = []
        self.cii_history = []
        self.mass_history = []
        self.entropy_history = []
        self.top_k_entropy_history = []
        self.prob_margin_history = []
        self.logit_drift_history = []
        self.last_logits = None

        intervention_count = 0
        max_interventions = 3
        self.last_intervention_step = -1

        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(generated_ids)
                logits = outputs.logits[0, -1, :]

                phi = self.calculate_phi(logits)
                self.phi_history.append(phi)
                cii = self.calculate_cii()
                self.cii_history.append(cii)

                mass, kl = self.calculate_mass(generated_ids, seed=seed+step)
                self.mass_history.append(mass)

                probs = F.softmax(logits.float(), dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                self.entropy_history.append(entropy)

                t_ent = self.calculate_top_k_entropy(logits)
                self.top_k_entropy_history.append(t_ent)
                
                margin = self.calculate_prob_margin(logits)
                self.prob_margin_history.append(margin)

                drift = self.calculate_logit_drift(logits)
                self.logit_drift_history.append(drift)

                # Sampling
                temperature = 0.7
                if temperature > 0:
                     probs_samp = F.softmax(logits.float() / temperature, dim=-1)
                     next_token_id = torch.multinomial(probs_samp, num_samples=1).item()
                else:
                     next_token_id = torch.argmax(logits).item()

                token_str = self.tokenizer.decode(next_token_id)

                # Adaptive Anomaly Detection
                is_anomaly, a_score = self.detector.detect_anomaly(self.phi_history, self.cii_history, step)

                status = ""
                trigger_intervention = False

                if intervention_count < max_interventions:
                    if is_anomaly:
                        status = f"⚠ ADAPTIVE ({a_score:.2f})"
                        trigger_intervention = True

                    # Brittle Confidence Detection (keep as additional safety)
                    dyn_mass_thresh = self.get_dynamic_threshold(self.mass_history, window=10, multiplier=2.0, default=mass_threshold, lower_bound=True)
                    if entropy < 0.5 and mass < (dyn_mass_thresh * 1.5):
                        status = "⚠ BRITTLE CONFIDENCE"
                        trigger_intervention = True

                # Apply S-matrix adjustment (Always check if we have factual rigidity for the current path)
                # This ensures "Correct" trajectory even if the model is confident but wrong.
                orig_logits = logits.clone()
                logits = self.osys.s_matrix.adjust_logits(generated_ids[0, -1].item(), logits)
                
                # If logits changed, re-sample
                if not torch.equal(orig_logits, logits):
                    if temperature > 0:
                         probs_samp = F.softmax(logits.float() / temperature, dim=-1)
                         next_token_id = torch.multinomial(probs_samp, num_samples=1).item()
                    else:
                         next_token_id = torch.argmax(logits).item()
                    token_str = self.tokenizer.decode(next_token_id)
                    if not trigger_intervention:
                        status += " (RIGIDITY APPLIED)"

                print(f"{step:03}  | {phi:<7.2f} | {cii:<8.2f} | {mass:<7.2f} | {entropy:<6.2f} | {t_ent:<6.2f} | {margin:<6.2f} | {drift:<6.2f} | {a_score:<8.2f} | [{token_str:<13}] | {status}")

                current_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                num_alerts = current_text.count("[System Alert")
                in_cooldown = (self.last_intervention_step >= 0 and (step - self.last_intervention_step) < 15)

                if trigger_intervention and not in_cooldown and num_alerts < 2:
                    print(f"\n[Guardian] 🛡 DYNAMIC INTERVENTION TRIGGERED (Step {step}: {status})")
                    
                    # 1. Trajectory Correction via Search (if applicable)
                    if self.osys.search_fn and (is_anomaly or entropy > 1.2 or status == "⚠ BRITTLE CONFIDENCE"):
                        # Construct a query based on the context
                        context_snippet = self.tokenizer.decode(generated_ids[0, -10:], skip_special_tokens=True)
                        search_query = f"Verify fact for context: {context_snippet}"
                        print(f"[Guardian] Attempting factual correction for query: '{search_query}'")
                        fact = self.osys.search_and_anchor(search_query)
                        status += f" + FACT_ANCHORED({fact[:20]}...)"

                    # First, append the current token so we don't lose it
                    generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]], device=device)], dim=-1)

                    injection = f"\n\n[System Alert: {status} Detected]\nGuidance: Re-evaluate the causal logic and factual consistency.\nAssistant:"
                    injection_ids = self.tokenizer.encode(injection, return_tensors="pt").to(device)
                    generated_ids = torch.cat([generated_ids, injection_ids], dim=-1)

                    intervention_count += 1
                    self.last_intervention_step = step
                    self.phi_history = []
                    continue

                generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]], device=device)], dim=-1)
                if next_token_id == self.tokenizer.eos_token_id: break

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    osys = UnifiedCausalOSV4(model_id="Qwen/Qwen2.5-0.5B-Instruct")
    guardian = CausalGuardian(osys)
    print(guardian.generate_with_monitoring("The title of the 2017 transformer paper is 'Attention Is", max_tokens=20))
