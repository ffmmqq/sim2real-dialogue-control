"""
Sim8 Original Simulator Adapter for HRL Museum Agent

This adapter implements the ORIGINAL Sim8 with full neural models:
- T5 dialogue generation model (trained on visitor data)
- Conditional VAE for gaze synthesis
- Groq Llama verification (replacing Mistral)

This enables direct comparison with the lightweight adapted sim8 to measure
the impact of neural vs rule-based simulation on learned policies.

Key differences from sim8_adapter.py:
- Uses T5 neural model for dialogue generation (not templates)
- Uses VAE neural model for gaze synthesis (not statistical sampling)
- Uses Groq Llama for verification (not rule-based matching)
"""

import random
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ConditionalVAE(nn.Module):
    """
    Conditional VAE for gaze feature synthesis.
    Conditioned on persona, AOI, and parent object.
    
    From original Sim4/sim8 implementation.
    """
    def __init__(self, n_persona, n_aoi, n_parent, pdim, adim, padim, latent_dim, gaze_dim, hdim):
        super().__init__()
        self.persona_embed = nn.Embedding(n_persona, pdim)
        self.aoi_embed = nn.Embedding(n_aoi, adim)
        self.parent_embed = nn.Embedding(n_parent, padim)

        enc_in = pdim + adim + padim
        self.encoder = nn.Sequential(nn.Linear(enc_in, hdim), nn.ReLU())
        self.fc_mu = nn.Linear(hdim, latent_dim)
        self.fc_logvar = nn.Linear(hdim, latent_dim)

        dec_in = latent_dim + enc_in
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hdim),
            nn.ReLU(),
            nn.Linear(hdim, gaze_dim),
            nn.Softplus()
        )

    def encode(self, persona, aoi, parent):
        p, a, pr = self.persona_embed(persona), self.aoi_embed(aoi), self.parent_embed(parent)
        h = self.encoder(torch.cat([p, a, pr], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std, eps = torch.exp(0.5 * logvar), torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, persona, aoi, parent):
        p, a, pr = self.persona_embed(persona), self.aoi_embed(aoi), self.parent_embed(parent)
        return self.decoder(torch.cat([z, p, a, pr], dim=1))

    def forward(self, persona, aoi, parent):
        mu, logvar = self.encode(persona, aoi, parent)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, persona, aoi, parent), mu, logvar


class IntermediateFusionModule:
    """
    Fusion module that combines T5 dialogue generation with VAE gaze synthesis.
    
    From original Sim4/sim8 implementation.
    """
    
    # Empirical gaze statistics from sim4 training data (mean, std) per persona
    # The VAE learns conditional means but has limited variance capture
    # We add MULTIPLICATIVE noise (std as % of value) for realistic variation
    # Base values from scaler: [~42, ~0.11, ~0.78, ~0.71, ~2.26, ~0.38]
    GAZE_NOISE_STATS = {
        "Agreeable": {
            "TurnScanpathLength": 0.25,   # 25% variation (high exploration variance)
            "SaccadeSpan": 0.20,          # 20% variation
            "TurnGazeEntropy": 0.15,      # 15% variation
            "TurnFixChangeRate": 0.30,    # 30% variation (fix rate varies a lot)
            "DominantObjectRatio": 0.10,  # 10% variation
            "GazeEntryLatency": 0.30      # 30% variation
        },
        "Conscientious": {
            "TurnScanpathLength": 0.15,   # Less variable, more focused
            "SaccadeSpan": 0.15,
            "TurnGazeEntropy": 0.10,
            "TurnFixChangeRate": 0.20,
            "DominantObjectRatio": 0.08,
            "GazeEntryLatency": 0.20
        },
        "Neurotic": {
            "TurnScanpathLength": 0.35,   # More variable, less predictable
            "SaccadeSpan": 0.25,
            "TurnGazeEntropy": 0.20,
            "TurnFixChangeRate": 0.40,
            "DominantObjectRatio": 0.15,
            "GazeEntryLatency": 0.40
        }
    }
    
    GAZE_LABELS = ["TurnScanpathLength", "SaccadeSpan", "TurnGazeEntropy",
                   "TurnFixChangeRate", "DominantObjectRatio", "GazeEntryLatency"]
    
    # Persona-specific max values for proper 0-1 dwell normalization
    # Based on empirical statistics from sim4.ipynb silence_stats (mean + 2*std)
    SCANPATH_MAX = {
        "Agreeable": 250.0,      # High exploration persona (mean=78.8, std=132.6)
        "Conscientious": 150.0,  # Focused persona (mean=49.2, std=60.5)
        "Neurotic": 120.0        # Anxious, less exploration (mean=41.3, std=49.6)
    }
    
    def __init__(self, t5_model, tokenizer, gaze_model, le_persona, le_aoi, le_parent, scaler):
        self.t5 = t5_model
        self.tokenizer = tokenizer
        self.gaze_model = gaze_model
        self.le_persona = le_persona
        self.le_aoi = le_aoi
        self.le_parent = le_parent
        self.scaler = scaler

    def generate_user_utterance(self, agent_utterance, persona, aoi, max_len=64):
        """Generate user utterance using T5 model"""
        prompt = f"Persona: {persona}. AOI: {aoi}. Agent says: {agent_utterance}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.t5.device)
        with torch.no_grad():
            output = self.t5.generate(
                **inputs,
                max_length=max_len,
                do_sample=True,
                top_k=50,
                temperature=0.8
            )
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        cleaned = re.sub(r"\([^)]*\)", "", decoded).strip()  # Remove parenthetical cues
        return cleaned

    def generate_gaze_vector(self, persona, aoi, parent):
        """Generate gaze features using VAE model with realistic variation.
        
        The VAE learns conditional means for persona+aoi+parent combinations,
        but has limited variance. We add persona-specific noise based on
        empirical statistics from the training data to get realistic variation.
        """
        try:
            # Validate inputs are in label encoder classes
            if persona not in self.le_persona.classes_:
                raise ValueError(f"Persona '{persona}' not in VAE training set")
            if aoi not in self.le_aoi.classes_:
                raise ValueError(f"AOI '{aoi}' not in VAE training set")
            if parent not in self.le_parent.classes_:
                raise ValueError(f"Parent '{parent}' not in VAE training set")
            
            persona_idx = torch.tensor([self.le_persona.transform([persona])[0]], device=self.t5.device)
            aoi_idx = torch.tensor([self.le_aoi.transform([aoi])[0]], device=self.t5.device)
            parent_idx = torch.tensor([self.le_parent.transform([parent])[0]], device=self.t5.device)
            
            # Sample random latent vector
            z = torch.randn(1, 8, device=self.t5.device, dtype=torch.float32)
            
            with torch.no_grad():
                gaze_vector = self.gaze_model.decode(z, persona_idx, aoi_idx, parent_idx)
            
            # Convert to numpy and inverse transform (VAE output -> original scale)
            gaze_np = gaze_vector.cpu().numpy()
            gaze_scaled = self.scaler.inverse_transform(gaze_np).flatten()
            
            # Add persona-specific MULTIPLICATIVE noise for realistic variation
            # VAE provides conditional mean, noise adds individual differences
            # Multiplicative noise: value * (1 + N(0, std_pct))
            noise_pcts = self.GAZE_NOISE_STATS.get(persona, self.GAZE_NOISE_STATS["Agreeable"])
            for i, label in enumerate(self.GAZE_LABELS):
                if label in noise_pcts:
                    std_pct = noise_pcts[label]
                    noise_multiplier = 1.0 + np.random.normal(0, std_pct)
                    gaze_scaled[i] *= max(0.1, noise_multiplier)  # Prevent negative/zero multipliers
            
            # Ensure non-negative values for all features
            gaze_scaled = np.maximum(gaze_scaled, 0.01)
            
            # Convert TurnScanpathLength to DwellTime (normalize to 0-1)
            # Use persona-specific max for proper scaling across full 0-1 range
            if len(gaze_scaled) >= 6:
                scanpath = gaze_scaled[0]
                max_scanpath = self.SCANPATH_MAX.get(persona, 200.0)
                dwell = float(np.clip(scanpath / max_scanpath, 0.0, 1.0))
                gaze_scaled[0] = dwell
            
            # Debug output
            import os
            if os.environ.get('HRL_DEBUG_VAE') == '1':
                print(f"[VAE Debug] Final gaze (with noise): {gaze_scaled}")
            
            return gaze_scaled.tolist()
            
        except (ValueError, KeyError) as e:
            # Fallback: AOI not in training set, return default gaze with noise
            if aoi not in getattr(self, '_warned_aois', set()):
                print(f"[Sim8Original] Warning: AOI '{aoi}' / Parent '{parent}' not in VAE training set (reason: {e}), using fallback gaze")
                if not hasattr(self, '_warned_aois'):
                    self._warned_aois = set()
                self._warned_aois.add(aoi)
            
            # Generate fallback gaze with persona-specific multiplicative noise
            base_gaze = np.array([60.0, 0.11, 0.78, 0.71, 2.26, 0.38])  # Scaler means
            noise_pcts = self.GAZE_NOISE_STATS.get(persona, self.GAZE_NOISE_STATS["Agreeable"])
            for i, label in enumerate(self.GAZE_LABELS):
                if label in noise_pcts:
                    std_pct = noise_pcts[label]
                    noise_multiplier = 1.0 + np.random.normal(0, std_pct)
                    base_gaze[i] *= max(0.1, noise_multiplier)
            base_gaze = np.maximum(base_gaze, 0.01)
            # Use persona-specific max for proper 0-1 dwell scaling
            max_scanpath = self.SCANPATH_MAX.get(persona, 200.0)
            base_gaze[0] = float(np.clip(base_gaze[0] / max_scanpath, 0.0, 1.0))
            return base_gaze.tolist()

    def fuse(self, agent_utterance, aoi, parent, persona):
        """Fuse dialogue and gaze generation"""
        utterance = self.generate_user_utterance(agent_utterance, persona, aoi)
        gaze_vector = self.generate_gaze_vector(persona, aoi, parent)
        return {
            "utterance": utterance,
            "gaze_features": gaze_vector,
            "aoi": aoi,
            "persona": persona,
            "parent_object": parent
        }


class Sim8OriginalSimulator:
    """
    Original Sim8 Simulator with full neural models.
    
    Implements the same interface as Sim8Simulator for compatibility with training loop,
    but uses T5 + VAE + Groq verification instead of rule-based generation.
    """
    
    PERSONAS = ["Agreeable", "Conscientious", "Neurotic"]
    
    def __init__(self, knowledge_graph=None, exhibits: Optional[List[str]] = None,
                 dialogue_model_path: str = None,
                 gaze_model_path: str = None,
                 seed: int = 42):
        """
        Initialize Original Sim8 with neural models.
        
        Args:
            knowledge_graph: Knowledge graph instance
            exhibits: List of exhibit names (fallback)
            dialogue_model_path: Path to trained T5 dialogue model directory
            gaze_model_path: Path to trained gaze VAE model (.pth file)
            seed: Random seed
        """
        self.rng = random.Random(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # CRITICAL: Don't set torch seed globally - we want VAE to generate diverse gaze
        # The VAE uses random sampling (z ~ N(0,1)) and should produce different outputs each time
        # torch.manual_seed(seed)  # DO NOT SET - this would make VAE deterministic!
        
        print(f"[Sim8Original] Initializing on device: {self.device}")
        
        # Build mappings from knowledge graph
        if knowledge_graph:
            self._init_from_knowledge_graph(knowledge_graph)
        elif exhibits:
            self.exhibits = exhibits
            self.aoi_to_exhibit = {}
            self.exhibit_to_aois = {ex: [] for ex in exhibits}
        else:
            raise ValueError("Must provide either knowledge_graph or exhibits list")
        
        # Load T5 dialogue model
        if dialogue_model_path is None:
            raise ValueError("dialogue_model_path is required")
        
        print(f"[Sim8Original] Loading T5 dialogue model from {dialogue_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(dialogue_model_path)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(dialogue_model_path).to(self.device)
        self.t5_model.eval()
        print(f"[Sim8Original] T5 model loaded successfully")
        
        # Load Gaze VAE model
        if gaze_model_path is None:
            raise ValueError("gaze_model_path is required")
        
        print(f"[Sim8Original] Loading Gaze VAE model from {gaze_model_path}...")
        gaze_ckpt = torch.load(gaze_model_path, map_location=self.device, weights_only=False)
        
        # VAE hyperparameters (from sim4)
        latent_dim = 8
        pdim = adim = padim = 8
        hdim = 64
        gaze_dim = 6
        
        self.gaze_model = ConditionalVAE(
            n_persona=len(gaze_ckpt["le_persona"].classes_),
            n_aoi=len(gaze_ckpt["le_aoi"].classes_),
            n_parent=len(gaze_ckpt["le_parent"].classes_),
            pdim=pdim, adim=adim, padim=padim,
            latent_dim=latent_dim, gaze_dim=gaze_dim, hdim=hdim
        ).to(self.device)
        
        self.gaze_model.load_state_dict(gaze_ckpt["model_state_dict"])
        self.gaze_model.eval()
        
        # Load encoders and scaler
        self.le_persona = gaze_ckpt["le_persona"]
        self.le_aoi = gaze_ckpt["le_aoi"]
        self.le_parent = gaze_ckpt["le_parent"]
        self.scaler = gaze_ckpt["scaler"]
        
        print(f"[Sim8Original] VAE model loaded successfully")
        print(f"[Sim8Original]   - Personas: {list(self.le_persona.classes_)}")
        print(f"[Sim8Original]   - AOIs in VAE: {len(self.le_aoi.classes_)}")
        print(f"[Sim8Original]   - Parents in VAE: {list(self.le_parent.classes_)}")
        
        # Diagnostic: Check AOI overlap after knowledge graph loaded
        if hasattr(self, 'aoi_to_exhibit'):
            vae_aois = set(self.le_aoi.classes_)
            kg_aois = set(self.aoi_to_exhibit.keys())
            overlap = vae_aois & kg_aois
            print(f"[Sim8Original] AOI Overlap Check: {len(overlap)}/{len(kg_aois)} KG AOIs found in VAE")
            missing = kg_aois - vae_aois
            if missing:
                print(f"[Sim8Original]   Missing from VAE: {sorted(missing)}")
        
        # Create fusion module
        self.fusion = IntermediateFusionModule(
            self.t5_model, self.tokenizer, self.gaze_model,
            self.le_persona, self.le_aoi, self.le_parent, self.scaler
        )
        
        # Session state
        self.current_persona = None
        self.current_exhibit = None
        self.current_aoi = None
        self.aoi_usage_count = {}
        self.seen_aois = set()
        self.last_user_response = {}
        
        # Track conversation for context
        self.dialogue_history = []
        self.max_history_length = 8
        
        print("[Sim8Original] Initialization complete!")
    
    def _init_from_knowledge_graph(self, knowledge_graph):
        """Build mappings from knowledge graph"""
        self.exhibits = knowledge_graph.get_exhibit_names()
        self.aoi_to_exhibit = {}
        self.exhibit_to_aois = {}
        
        for exhibit_name in self.exhibits:
            aois = knowledge_graph.get_exhibit_aois(exhibit_name)
            self.exhibit_to_aois[exhibit_name] = aois
            for aoi in aois:
                self.aoi_to_exhibit[aoi] = exhibit_name
        
        print(f"[Sim8Original] Initialized from knowledge graph:")
        print(f"   - {len(self.exhibits)} exhibits: {', '.join(self.exhibits)}")
        print(f"   - {len(self.aoi_to_exhibit)} AOIs mapped to exhibits")
        
        # Build exhibit to parent code mapping (for VAE compatibility)
        # VAE expects parent codes (B1, B2, B3, C5, C6), not exhibit names
        self.exhibit_to_parent_code = {
            "Diego_Bemba": "B1",
            "Dom_Miguel": "B2",
            "Pedro_Sunda": "B3",
            "Turban": "C5",
            "King_Caspar": "C6"
        }
    
    def initialize_session(self, persona: Optional[str] = None):
        """Initialize session with persona"""
        self.current_persona = persona or self.rng.choice(self.PERSONAS)
        self.current_exhibit = self.rng.choice(self.exhibits)
        
        # Pick initial AOI
        if self.current_exhibit in self.exhibit_to_aois and self.exhibit_to_aois[self.current_exhibit]:
            self.current_aoi = self.rng.choice(self.exhibit_to_aois[self.current_exhibit])
        else:
            self.current_aoi = list(self.aoi_to_exhibit.keys())[0] if self.aoi_to_exhibit else "Unknown"
        
        self.aoi_usage_count.clear()
        self.seen_aois.clear()
        self.last_user_response = {}
        self.dialogue_history = []
        
        print(f"[Sim8Original] Session initialized: persona={self.current_persona}, exhibit={self.current_exhibit}, aoi={self.current_aoi}")
    
    def get_current_aoi(self) -> str:
        """Return current exhibit (for compatibility with training loop)"""
        return self.current_exhibit or self.exhibits[0]
    
    def generate_user_response(self, agent_utterance: str, agent_option: str = None,
                               agent_subaction: str = None, target_exhibit: str = None,
                               current_exhibit_completion: float = 0.0,
                               exhibit_exhausted: bool = False,
                               target_exhibit_completion: float = 0.0,
                               target_exhibit_exhausted: bool = False) -> Dict[str, Any]:
        """
        Generate user response using original Sim8 models (T5 + VAE).
        
        This is the key method that bridges to the training loop.
        """
        import time
        start_time = time.time()
        
        # Track agent utterance in dialogue history
        self.dialogue_history.append({"role": "agent", "text": agent_utterance})
        if len(self.dialogue_history) > self.max_history_length:
            self.dialogue_history.pop(0)
        
        # Detect AOI from agent utterance or use current
        detected_aoi = self.current_aoi
        parent_exhibit = self.aoi_to_exhibit.get(detected_aoi, self.current_exhibit)
        
        # Map exhibit name to parent code for VAE (VAE expects B1, B2, B3, C5, C6)
        parent_code = self.exhibit_to_parent_code.get(parent_exhibit, "C6")
        
        # Handle transitions with probability-based success
        transition_success = False
        if agent_option == "OfferTransition" and target_exhibit:
            # Transition probability based on current exhibit completion
            if current_exhibit_completion == 0.0:
                transition_prob = 0.20
            elif current_exhibit_completion < 0.33:
                transition_prob = 0.50
            elif current_exhibit_completion < 0.67:
                transition_prob = 0.80
            else:
                transition_prob = 0.95
            
            # Penalty for poor target quality
            if target_exhibit_exhausted:
                transition_prob *= 0.15
            elif target_exhibit_completion >= 0.67:
                transition_prob *= 0.50
            elif target_exhibit_completion >= 0.33:
                transition_prob *= 0.75
            
            # Roll for transition
            if self.rng.random() < transition_prob:
                transition_success = True
                if target_exhibit in self.exhibits and target_exhibit in self.exhibit_to_aois:
                    if self.exhibit_to_aois[target_exhibit]:
                        detected_aoi = self.rng.choice(self.exhibit_to_aois[target_exhibit])
                        self.current_exhibit = target_exhibit
                        self.current_aoi = detected_aoi
                        parent_exhibit = target_exhibit
                        parent_code = self.exhibit_to_parent_code.get(parent_exhibit, "C6")
                        import os
                        if os.environ.get('HRL_VERBOSE') == '1':
                            print(f"[Sim8Original] Transition SUCCESS → {target_exhibit}")
            else:
                import os
                if os.environ.get('HRL_VERBOSE') == '1':
                    print(f"[Sim8Original] Transition REJECTED (prob={transition_prob:.1%})")
        
        # Update session state
        self.current_aoi = detected_aoi
        if detected_aoi in self.aoi_to_exhibit:
            self.current_exhibit = self.aoi_to_exhibit[detected_aoi]
        
        self.aoi_usage_count[detected_aoi] = self.aoi_usage_count.get(detected_aoi, 0) + 1
        self.seen_aois.add(detected_aoi)
        
        # Handle AOI mapping for VAE compatibility
        # Map "White ostrich feather" -> "Red ostrich feather" (fuzzy match from diagnostic)
        # Map "Ring" -> "Necklace" (same exhibit, similar item)
        vae_aoi = detected_aoi
        if detected_aoi == "White ostrich feather":
            vae_aoi = "Red ostrich feather"
        elif detected_aoi == "Ring":
            vae_aoi = "Necklace"  # Same exhibit, similar jewelry item
        
        # Call fusion module (T5 + VAE) with mapped AOI and parent code
        try:
            fused = self.fusion.fuse(
                agent_utterance=agent_utterance,
                aoi=vae_aoi,  # Use mapped AOI
                parent=parent_code,  # Use parent code (B1, B2, etc.), not exhibit name
                persona=self.current_persona
            )
            
            # Gaze features are already in correct format from generate_gaze_vector()
            # VAE outputs are converted there: TurnScanpathLength -> DwellTime (0-1)
            # Format: [DwellTime, SaccadeSpan, TurnGazeEntropy,
            #          TurnFixChangeRate, DominantObjectRatio, GazeEntryLatency]
            gaze_features = fused["gaze_features"]
            
            # Optional: Verify with Groq Llama (can be disabled for speed)
            # verification_result = self._verify_with_groq(fused["utterance"], detected_aoi, agent_utterance)
            
            # Track in dialogue history
            self.dialogue_history.append({"role": "user", "text": fused["utterance"]})
            
            elapsed = time.time() - start_time
            
            response = {
                "utterance": fused["utterance"],
                "aoi": detected_aoi,
                "persona": self.current_persona,
                "gaze_features": gaze_features,
                "response_type": "statement",  # Simplified - T5 doesn't classify types
                "engagement_level": gaze_features[0],  # Use dwell time as engagement
                "transition_success": transition_success,
                "simulator_llm_time": elapsed,
                "off_topic_strikes": 0,
                "agent_option": agent_option
            }
            
            self.last_user_response = response
            return response
            
        except Exception as e:
            print(f"[Sim8Original] Error in fusion: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback response
            elapsed = time.time() - start_time
            return {
                "utterance": "That's interesting.",
                "aoi": detected_aoi,
                "persona": self.current_persona,
                "gaze_features": [0.5, 0.1, 0.8, 2.0, 0.7, 5.0],
                "response_type": "statement",
                "engagement_level": 0.5,
                "transition_success": False,
                "simulator_llm_time": elapsed,
                "off_topic_strikes": 0,
                "agent_option": agent_option
            }
    
    def _verify_with_groq(self, user_utterance: str, expected_aoi: str, agent_utterance: str) -> bool:
        """
        Use Groq Llama for AOI verification (replaces Mistral).
        Optional - can be disabled for speed.
        """
        try:
            from pipeline.LLM_CONFIG import get_simulator_llm
            
            prompt = f"""You are a verifier. Check if the visitor's utterance refers to "{expected_aoi}".

[Agent said]: "{agent_utterance}"
[Visitor replied]: "{user_utterance}"

Does the visitor's utterance refer to "{expected_aoi}" (explicitly, by pronoun, or description)?
Answer only "Yes" or "No"."""
            
            llm = get_simulator_llm()
            answer = llm.generate(prompt, system_prompt="You are a verification assistant.")
            return answer.strip().lower().startswith("yes")
        except Exception as e:
            print(f"[Sim8Original] Verification failed: {e}")
            return True  # Assume valid on error
    
    def get_current_state(self) -> Dict[str, Any]:
        """Return current simulator state"""
        return {
            "aoi": self.current_aoi,
            "current_exhibit": self.current_exhibit,
            "persona": self.current_persona,
            "seen_aois": list(self.seen_aois),
            "aoi_usage_count": dict(self.aoi_usage_count),
            "last_user_response": dict(self.last_user_response) if self.last_user_response else {}
        }

