"""
LLM Handler for Museum Dialogue Generation

This module provides a flexible interface for using LLMs. It supports multiple backends:

1. Groq API (recommended): Fast inference with Llama models
2. OpenRouter: OpenAI-compatible API with provider routing
3. Local OpenAI-compatible server (for example vLLM serving Llama 3.1)
4. Hugging Face: Local models (Phi-2, TinyLLaMA, Mistral, FLAN-T5, GPT-2)
5. Mistral API: Free tier available

Default: Groq API with Llama 3.1 8B (fast and efficient)
"""

import os
import json
import textwrap
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum


class LLMBackend(Enum):
    """Available LLM backends."""
    GROQ = "groq"
    OPENROUTER = "openrouter"
    LOCAL_OPENAI = "local_openai"
    HUGGINGFACE = "huggingface"
    MISTRAL_API = "mistral_api"


class LLMCriticalError(Exception):
    """Raised when LLM encounters a critical, unrecoverable error (e.g., spend limit, auth failure)."""
    pass


class FreeLLMHandler:
    """
    Handler for LLM inference.
    
    This provides a unified interface for different LLM backends.
    Default: Groq API with Llama 3.1 8B for fast, efficient inference.
    
    Available Groq models:
    - llama-3.1-8b: Default, fast and cost-effective (8B)
    - llama-3.1: Higher quality (70B)
    - llama-3.3: Latest high quality (70B)
    
    For HuggingFace (local models):
    - phi2, tinyllama, neural-chat, mistral
    """
    
    def __init__(
        self,
        backend: str = "groq",
        model_name: str = "llama-3.1-8b",
        temperature: float = 0.7,
        max_tokens: int = 250,
        device: Optional[str] = None
    ):
        """
        Initialize LLM handler.
        
        Args:
            backend: LLM backend ('groq', 'openrouter', 'local_openai', 'huggingface', 'mistral_api')
            model_name: Model name (default: 'llama-3.1-8b' for Groq)
                       Options: llama-3.1-8b, llama-3.3, llama-3.1 (Groq)
                       Or local models: phi2, tinyllama, neural-chat, mistral (HuggingFace)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            device: Device for local models ('cpu', 'cuda')
        """
        self.backend = LLMBackend(backend.lower())
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device or ('cuda' if self._has_cuda() else 'cpu')
        
        # Check for fast mode
        import os
        self.fast_mode = os.environ.get('HRL_FAST_MODE') == '1'
        
        # Critical error tracking
        self.critical_error_count = 0
        self.max_critical_errors = 3  # Allow a few errors before failing
        self.last_critical_error = None
        
        # Retry configuration for connection errors
        self.max_retries = 100  # Maximum number of retry attempts
        self.retry_delay_initial = 2.0  # Initial delay in seconds
        self.retry_delay_max = 60.0  # Maximum delay in seconds
        self.retry_delay_multiplier = 1.5  # Exponential backoff multiplier
        
        # Connection health tracking
        self.total_api_calls = 0
        self.total_retries = 0
        self.total_api_time = 0.0  # Total time spent in API calls (including retries)
        self.consecutive_failures = 0
        self.max_consecutive_failures = 0
        
        if not self.fast_mode:
            # Initialize backend normally
            self._initialize_backend()
        else:
            print("[FAST MODE] Skipping LLM backend initialization")
        
    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _initialize_backend(self):
        """Initialize the selected LLM backend."""
        if self.backend == LLMBackend.GROQ:
            self._initialize_groq()
        elif self.backend == LLMBackend.OPENROUTER:
            self._initialize_openrouter()
        elif self.backend == LLMBackend.LOCAL_OPENAI:
            self._initialize_local_openai()
        elif self.backend == LLMBackend.HUGGINGFACE:
            self._initialize_huggingface()
        elif self.backend == LLMBackend.MISTRAL_API:
            self._initialize_mistral_api()

    def _load_api_key_from_file(self, env_var: str, filename: str) -> Optional[str]:
        """Load an API key from env or a project-root key file."""
        api_key = os.environ.get(env_var)
        if api_key:
            return api_key

        key_file = Path(__file__).parent.parent.parent / filename
        if not key_file.exists():
            return None

        try:
            api_key = key_file.read_text().strip()
            os.environ[env_var] = api_key
            print(f"[OK] Loaded {env_var} from {filename}")
            return api_key
        except Exception as e:
            print(f"[WARN] Failed to read {filename}: {e}")
            return None
    
    def _initialize_huggingface(self):
        """Initialize Hugging Face backend."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print(f"Loading Hugging Face model: {self.model_name}...")
            
            # Map common names to HF model IDs
            model_map = {
                "phi2": "microsoft/phi-2",
                "phi": "microsoft/phi-2",
                "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "tiny-llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "neural-chat": "Intel/neural-chat-7b-v3-1",
                "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
                "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
                "flan-t5": "google/flan-t5-base",
                "gpt2": "gpt2",
                "gpt2-medium": "gpt2-medium"
            }
            
            model_id = model_map.get(self.model_name.lower(), self.model_name)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map='auto' if self.device == 'cuda' else None
            )
            
            if self.device == 'cpu':
                self.model = self.model.to('cpu')
            
            print(f"[OK] Loaded {model_id} on {self.device}")
            self.hf_available = True
            
        except ImportError:
            print("[WARN] transformers not installed: pip install transformers torch")
            self.hf_available = False
        except Exception as e:
            print(f"[WARN] Error loading Hugging Face model: {e}")
            self.hf_available = False
    
    def _initialize_mistral_api(self):
        """Initialize Mistral API backend."""
        try:
            if "MISTRAL_API_KEY" not in os.environ:
                print("[WARN] MISTRAL_API_KEY not set in environment")
                self.mistral_api_available = False
            else:
                from mistralai.client import MistralClient
                self.mistral_client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
                print("[OK] Mistral API client initialized")
                self.mistral_api_available = True
        except ImportError:
            print("[WARN] mistralai not installed: pip install mistralai")

    def _initialize_groq(self):
        """Initialize Groq API backend."""
        try:
            api_key = self._load_api_key_from_file("GROQ_API_KEY", "groq_key.txt")
            
            if not api_key:
                print("[WARN] GROQ_API_KEY not set in environment and groq_key.txt not found")
                print("  Get free API key: https://console.groq.com/")
                print("  Or create groq_key.txt in project root with your API key")
                self.groq_available = False
            else:
                from groq import Groq
                self.groq_client = Groq(api_key=api_key)
                print(f"[OK] Groq API client initialized")
                print(f"  Using model: {self.model_name}")
                self.groq_available = True
        except ImportError:
            print("[WARN] groq not installed: pip install groq")
            self.groq_available = False
        except Exception as e:
            print(f"[WARN] Error initializing Groq: {e}")
            self.groq_available = False

    def _initialize_openrouter(self):
        """Initialize OpenRouter backend."""
        try:
            api_key = self._load_api_key_from_file("OPENROUTER_API_KEY", "openrouter_key.txt")

            if not api_key:
                print("[WARN] OPENROUTER_API_KEY not set in environment and openrouter_key.txt not found")
                print("  Get API key: https://openrouter.ai/settings/keys")
                print("  Or create openrouter_key.txt in project root with your API key")
                self.openrouter_available = False
                return

            self.openrouter_api_key = api_key
            self.openrouter_base_url = os.environ.get(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1/chat/completions"
            )
            self.openrouter_provider_config = self._build_openrouter_provider_config()
            self.openrouter_available = True
            print("[OK] OpenRouter client configured")
            print(f"  Using model: {self.model_name}")
            if self.openrouter_provider_config:
                print(f"  Provider routing: {json.dumps(self.openrouter_provider_config)}")
        except Exception as e:
            print(f"[WARN] Error initializing OpenRouter: {e}")
            self.openrouter_available = False

    def _initialize_local_openai(self):
        """Configure an OpenAI-compatible local inference server such as vLLM."""
        base = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        self.local_openai_url = (
            base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        )
        self.local_openai_api_key = os.environ.get("LOCAL_LLM_API_KEY", "EMPTY")
        self.local_openai_available = True
        print("[OK] Local OpenAI-compatible LLM configured")
        print(f"  Endpoint: {self.local_openai_url}")
        print(f"  Model: {self.model_name}")

    def _build_openrouter_provider_config(self) -> Dict[str, Any]:
        """Build OpenRouter provider routing config from env vars."""
        provider: Dict[str, Any] = {}

        def parse_csv_env(name: str) -> List[str]:
            raw = os.environ.get(name, "")
            return [item.strip() for item in raw.split(",") if item.strip()]

        provider_only = parse_csv_env("OPENROUTER_PROVIDER_ONLY")
        provider_order = parse_csv_env("OPENROUTER_PROVIDER_ORDER")
        quantizations = parse_csv_env("OPENROUTER_QUANTIZATIONS")

        if provider_only:
            provider["only"] = provider_only
        elif provider_order:
            provider["order"] = provider_order

        if quantizations:
            provider["quantizations"] = quantizations

        allow_fallbacks = os.environ.get("OPENROUTER_ALLOW_FALLBACKS")
        if allow_fallbacks is not None:
            provider["allow_fallbacks"] = allow_fallbacks.strip().lower() in {"1", "true", "yes", "on"}

        require_parameters = os.environ.get("OPENROUTER_REQUIRE_PARAMETERS")
        if require_parameters is not None:
            provider["require_parameters"] = require_parameters.strip().lower() in {"1", "true", "yes", "on"}

        return provider
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text using the configured LLM backend.
        
        Args:
            prompt: User/input prompt
            system_prompt: System prompt (context)
            
        Returns:
            Generated text
        """
        # In fast mode, always use fallback
        if self.fast_mode:
            return self._fallback_response(prompt)
        
        if self.backend == LLMBackend.GROQ:
            return self._generate_groq(prompt, system_prompt)
        elif self.backend == LLMBackend.OPENROUTER:
            return self._generate_openrouter(prompt, system_prompt)
        elif self.backend == LLMBackend.LOCAL_OPENAI:
            return self._generate_local_openai(prompt, system_prompt)
        elif self.backend == LLMBackend.HUGGINGFACE:
            return self._generate_huggingface(prompt, system_prompt)
        elif self.backend == LLMBackend.MISTRAL_API:
            return self._generate_mistral_api(prompt, system_prompt)
        else:
            return self._fallback_response(prompt)
    
    def _generate_huggingface(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate using Hugging Face."""
        if not self.hf_available:
            return self._fallback_response(prompt)
        
        try:
            # Format prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Tokenize
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            if self.device == 'cuda':
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response (remove prompt)
            if generated_text.startswith(full_prompt):
                generated_text = generated_text[len(full_prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            print(f"[WARN] Hugging Face generation error: {e}")
            return self._fallback_response(prompt)
    
    def _generate_mistral_api(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate using Mistral API."""
        if not self.mistral_client:
            return self._fallback_response(prompt)
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.mistral_client.chat(
                model=self.model_name or "mistral-small",
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[WARN] Mistral API error: {e}")
            return self._fallback_response(prompt)

    def _test_openrouter_connection(self) -> bool:
        """Test OpenRouter with a minimal chat request."""
        if not getattr(self, "openrouter_available", False):
            return False

        try:
            import requests

            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5,
                "temperature": 0.1,
            }
            if self.openrouter_provider_config:
                payload["provider"] = self.openrouter_provider_config

            response = requests.post(
                self.openrouter_base_url,
                headers=self._openrouter_headers(),
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            return True
        except Exception:
            return False

    def _openrouter_headers(self) -> Dict[str, str]:
        """Build OpenRouter request headers."""
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        title = os.environ.get("OPENROUTER_APP_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        return headers

    def _generate_openrouter(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate using OpenRouter chat completions."""
        if not getattr(self, "openrouter_available", False):
            raise LLMCriticalError("OpenRouter API not available - cannot generate response")

        import requests
        import time

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.openrouter_provider_config:
            payload["provider"] = self.openrouter_provider_config

        call_start_time = time.time()
        retry_count = 0
        retry_delay = self.retry_delay_initial
        last_error = None
        self.total_api_calls += 1

        while retry_count < self.max_retries:
            try:
                response = requests.post(
                    self.openrouter_base_url,
                    headers=self._openrouter_headers(),
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()

                if "choices" not in data:
                    raise ValueError(self._format_openrouter_payload_error(response, data))

                message = data["choices"][0]["message"]["content"]
                if isinstance(message, list):
                    message = "".join(
                        part.get("text", "")
                        for part in message
                        if isinstance(part, dict)
                    )

                call_time = time.time() - call_start_time
                self.total_api_time += call_time
                self.total_retries += retry_count
                self.critical_error_count = 0
                self.consecutive_failures = 0

                if retry_count > 0:
                    print(f"[OK] OpenRouter connection restored after {retry_count} retries (took {call_time:.1f}s total)")

                return str(message).strip()

            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                critical_error_keywords = [
                    'spend_limit_reached',
                    'insufficient_quota',
                    'authentication_error',
                    'invalid_api_key',
                    'account_deactivated',
                    'payment required',
                    'credits'
                ]
                is_critical = any(keyword in error_str for keyword in critical_error_keywords)

                if is_critical:
                    self.critical_error_count += 1
                    self.last_critical_error = str(e)
                    print(f"\n{'='*80}")
                    print(f"CRITICAL LLM ERROR DETECTED ({self.critical_error_count}/{self.max_critical_errors})")
                    print(f"{'='*80}")
                    print(f"Error: {str(e)}")

                    if self.critical_error_count >= self.max_critical_errors:
                        print("\nCritical error threshold reached!")
                        print("Training will be stopped to save the model.")
                        print(f"{'='*80}\n")
                        raise LLMCriticalError(f"OpenRouter API critical error: {str(e)}")
                    retry_delay = min(retry_delay * 2, self.retry_delay_max)

                is_connection_error = any(keyword in error_str for keyword in [
                    'connection',
                    'timeout',
                    'network',
                    'unreachable',
                    'refused',
                    'reset',
                    '429',
                    'rate limit'
                ])

                if is_connection_error or not is_critical:
                    retry_count += 1
                    self.consecutive_failures += 1
                    self.max_consecutive_failures = max(self.max_consecutive_failures, self.consecutive_failures)

                    if retry_count == 1:
                        elapsed = time.time() - call_start_time
                        print(f"[WARN] OpenRouter error (after {elapsed:.1f}s): {str(e)}")
                        print("[RETRY] Waiting for connection to be restored...")
                        print(f"[STATS] API Health: {self.total_api_calls} calls, {self.total_retries} retries, {self.consecutive_failures} consecutive failures")

                    print(f"[RETRY] Attempt {retry_count}/{self.max_retries} - Testing connection...")
                    connection_ok = self._test_openrouter_connection()

                    if connection_ok:
                        print("[OK] Connection test passed, retrying request immediately...")
                        retry_delay = self.retry_delay_initial
                        continue

                    print(f"[WAIT] Connection not ready, waiting {retry_delay:.1f}s before retry...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * self.retry_delay_multiplier, self.retry_delay_max)
                else:
                    retry_count += 1
                    print(f"[RETRY] Attempt {retry_count}/{self.max_retries} - Waiting {retry_delay:.1f}s...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * self.retry_delay_multiplier, self.retry_delay_max)

        error_msg = f"OpenRouter API connection failed after {self.max_retries} retry attempts."
        if last_error:
            error_msg += f" Last error: {str(last_error)}"
        raise LLMCriticalError(error_msg)

    def _generate_local_openai(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate through a local OpenAI-compatible chat-completions endpoint."""
        if not getattr(self, "local_openai_available", False):
            raise LLMCriticalError("Local OpenAI-compatible LLM is not configured")

        import requests
        import time

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        local_top_p = os.environ.get("LOCAL_LLM_TOP_P")
        if local_top_p:
            payload["top_p"] = float(local_top_p)
        local_seed = os.environ.get("LOCAL_LLM_SEED")
        if local_seed:
            payload["seed"] = int(local_seed)

        headers = {
            "Authorization": f"Bearer {self.local_openai_api_key}",
            "Content-Type": "application/json",
        }
        last_error = None
        retry_delay = self.retry_delay_initial
        self.total_api_calls += 1
        call_start_time = time.time()
        for retry_count in range(self.max_retries):
            try:
                response = requests.post(
                    self.local_openai_url,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
                if "choices" not in data:
                    raise ValueError(f"Local LLM response missing choices: {data}")
                message = data["choices"][0]["message"]["content"]
                self.total_api_time += time.time() - call_start_time
                self.total_retries += retry_count
                self.consecutive_failures = 0
                return str(message).strip()
            except Exception as exc:
                last_error = exc
                self.consecutive_failures += 1
                self.max_consecutive_failures = max(
                    self.max_consecutive_failures, self.consecutive_failures
                )
                if retry_count == 0:
                    print(f"[WARN] Local LLM request failed: {exc}")
                if retry_count + 1 < self.max_retries:
                    time.sleep(retry_delay)
                    retry_delay = min(
                        retry_delay * self.retry_delay_multiplier, self.retry_delay_max
                    )
        raise LLMCriticalError(
            f"Local LLM failed after {self.max_retries} attempts. Last error: {last_error}"
        )

    def _format_openrouter_payload_error(self, response, data: Any) -> str:
        """Create a readable error message when OpenRouter returns a non-standard payload."""
        provider_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower().startswith("x-") or "provider" in key.lower()
        }

        if isinstance(data, dict):
            top_level_keys = sorted(data.keys())
            error_obj = data.get("error")
        else:
            top_level_keys = []
            error_obj = None

        payload_preview = textwrap.shorten(
            json.dumps(data, ensure_ascii=False, default=str),
            width=1200,
            placeholder=" ...",
        )

        return (
            "OpenRouter response missing 'choices'. "
            f"status={response.status_code}, "
            f"top_level_keys={top_level_keys}, "
            f"error={error_obj}, "
            f"provider_headers={provider_headers}, "
            f"payload={payload_preview}"
        )
    
    def _test_groq_connection(self) -> bool:
        """
        Test if Groq API connection is working by making a minimal API call.
        
        Returns:
            True if connection is working, False otherwise
        """
        if not self.groq_available or not hasattr(self, 'groq_client'):
            return False
        
        try:
            # Make a minimal test call
            test_response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7,
                max_tokens=5
            )
            return True
        except Exception:
            return False
    
    def _generate_groq(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate using Groq API with retry logic for connection errors."""
        if not self.groq_available:
            raise LLMCriticalError("Groq API not available - cannot generate response")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Map common model names to Groq model IDs (updated for current models)
        model_map = {
            "mistral": "llama-3.3-70b-versatile",  # Use Llama 3.3 as replacement
            "mixtral": "llama-3.3-70b-versatile",
            "llama": "llama-3.3-70b-versatile",
            "llama2": "llama-3.1-70b-versatile",
            "llama3": "llama-3.3-70b-versatile",
            "llama-3.3": "llama-3.3-70b-versatile",
            "llama-3.1": "llama-3.1-70b-versatile",
            "llama-3.1-8b": "llama-3.1-8b-instant",  # Cheaper 8B model!
        }
        groq_model = model_map.get(self.model_name.lower(), self.model_name)
        
        # Retry loop for connection errors
        import time
        call_start_time = time.time()
        retry_count = 0
        retry_delay = self.retry_delay_initial
        last_error = None
        self.total_api_calls += 1
        
        while retry_count < self.max_retries:
            try:
                response = self.groq_client.chat.completions.create(
                    model=groq_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Success - update stats
                call_time = time.time() - call_start_time
                self.total_api_time += call_time
                self.total_retries += retry_count
                self.critical_error_count = 0
                self.consecutive_failures = 0
                
                if retry_count > 0:
                    print(f"[OK] Groq API connection restored after {retry_count} retries (took {call_time:.1f}s total)")
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Detect CRITICAL errors that should stop training (no retry)
                critical_error_keywords = [
                    'spend_limit_reached',
                    'spend alert threshold',
                    'insufficient_quota',
                    'authentication_error',
                    'invalid_api_key',
                    'account_deactivated'
                ]
                
                is_critical = any(keyword in error_str for keyword in critical_error_keywords)
                
                if is_critical:
                    self.critical_error_count += 1
                    self.last_critical_error = str(e)
                    
                    print(f"\n{'='*80}")
                    print(f"CRITICAL LLM ERROR DETECTED ({self.critical_error_count}/{self.max_critical_errors})")
                    print(f"{'='*80}")
                    print(f"Error: {str(e)[:200]}")
                    
                    if self.critical_error_count >= self.max_critical_errors:
                        print(f"\nCritical error threshold reached!")
                        print(f"Training will be stopped to save the model.")
                        print(f"{'='*80}\n")
                        raise LLMCriticalError(f"Groq API critical error: {str(e)}")
                    else:
                        # For critical errors that haven't reached threshold, still retry
                        # but with longer delay
                        retry_delay = min(retry_delay * 2, self.retry_delay_max)
                
                # Check if it's a connection error (retry-able)
                is_connection_error = any(keyword in error_str for keyword in [
                    'connection',
                    'timeout',
                    'network',
                    'unreachable',
                    'refused',
                    'reset'
                ])
                
                if is_connection_error or not is_critical:
                    # Connection error - wait and retry
                    retry_count += 1
                    self.consecutive_failures += 1
                    self.max_consecutive_failures = max(self.max_consecutive_failures, self.consecutive_failures)
                    
                    if retry_count == 1:
                        elapsed = time.time() - call_start_time
                        print(f"[WARN] Groq API connection error (after {elapsed:.1f}s): {str(e)[:100]}")
                        print(f"[RETRY] Waiting for connection to be restored...")
                        print(f"[STATS] API Health: {self.total_api_calls} calls, {self.total_retries} retries, {self.consecutive_failures} consecutive failures")
                    
                    # Test connection before retrying
                    print(f"[RETRY] Attempt {retry_count}/{self.max_retries} - Testing connection...")
                    connection_ok = self._test_groq_connection()
                    
                    if connection_ok:
                        print(f"[OK] Connection test passed, retrying request immediately...")
                        retry_delay = self.retry_delay_initial  # Reset delay on successful test
                        # Don't increment retry_count here - we'll retry immediately in the next loop iteration
                        continue  # Retry immediately
                    else:
                        print(f"[WAIT] Connection not ready, waiting {retry_delay:.1f}s before retry...")
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * self.retry_delay_multiplier, self.retry_delay_max)
                        # Continue to next retry attempt (retry_count already incremented)
                else:
                    # Unknown error - treat as connection error and retry
                    retry_count += 1
                    if retry_count == 1:
                        print(f"[WARN] Groq API error: {str(e)[:100]}")
                        print(f"[RETRY] Waiting and retrying...")
                    
                    print(f"[RETRY] Attempt {retry_count}/{self.max_retries} - Waiting {retry_delay:.1f}s...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * self.retry_delay_multiplier, self.retry_delay_max)
        
        # If we've exhausted all retries, raise an error
        error_msg = f"Groq API connection failed after {self.max_retries} retry attempts."
        if last_error:
            error_msg += f" Last error: {str(last_error)}"
        raise LLMCriticalError(error_msg)
    
    def get_api_stats(self) -> Dict[str, Any]:
        """
        Get API call statistics.
        
        Returns:
            Dictionary with API health metrics
        """
        avg_time_per_call = self.total_api_time / self.total_api_calls if self.total_api_calls > 0 else 0
        retry_rate = self.total_retries / self.total_api_calls if self.total_api_calls > 0 else 0
        
        return {
            'total_calls': self.total_api_calls,
            'total_retries': self.total_retries,
            'total_time': self.total_api_time,
            'avg_time_per_call': avg_time_per_call,
            'retry_rate': retry_rate,
            'consecutive_failures': self.consecutive_failures,
            'max_consecutive_failures': self.max_consecutive_failures
        }
    
    def reset_episode_stats(self):
        """Reset per-episode statistics (call at start of each episode)."""
        # Keep cumulative stats, just reset consecutive failures
        self.consecutive_failures = 0
    
    def _fallback_response(self, prompt: str) -> str:
        """
        Fallback template-based response when LLM is unavailable.
        
        Provides museum dialogue responses based on prompt keywords.
        In fast mode, includes fact IDs for testing reward calculations.
        """
        prompt_lower = prompt.lower()
        
        # Extract action type from prompt
        if "explain" in prompt_lower or "tell" in prompt_lower or "fact" in prompt_lower:
            # For Explain actions, include fact IDs so novelty rewards work
            import random
            fact_ids = ["[KC_001]", "[KC_002]", "[KC_003]", "[KC_004]", "[KC_005]", "[KC_006]"]
            selected_facts = random.sample(fact_ids, 2)
            return f"This exhibit is fascinating. {selected_facts[0]} It represents wealth and status. {selected_facts[1]} The craftsmanship is remarkable."
        
        elif "ask" in prompt_lower and "opinion" in prompt_lower:
            return "What do you think about this piece? Does the craftsmanship appeal to you?"
        
        elif "ask" in prompt_lower and "memory" in prompt_lower:
            return "Do you remember the details we discussed about the previous exhibit?"
        
        elif "transition" in prompt_lower or "move" in prompt_lower:
            return "Shall we explore another fascinating piece nearby?"
        
        elif "conclude" in prompt_lower or "wrap" in prompt_lower:
            return "Thank you for visiting! I hope you've enjoyed learning about these remarkable exhibits."
        
        else:
            generic = [
                "That's an insightful observation about this exhibit.",
                "This piece has a rich history worth exploring.",
                "The cultural significance here is quite remarkable.",
                "There's much to appreciate in the details of this work.",
            ]
            import random
            return random.choice(generic)


# Global LLM handler instance
_llm_handler: Optional[FreeLLMHandler] = None


def get_llm_handler(
    backend: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> FreeLLMHandler:
    """
    Get global LLM handler instance (singleton).
    
    Args:
        backend: Override backend ('groq', 'openrouter', 'local_openai', 'huggingface', 'mistral_api')
        model_name: Override model name
        **kwargs: Additional arguments for FreeLLMHandler
        
    Returns:
        FreeLLMHandler instance
    """
    global _llm_handler
    
    # Check environment variables for configuration
    if backend is None:
        backend = os.environ.get("HRL_LLM_BACKEND", "groq")
    if model_name is None:
        model_name = os.environ.get("HRL_LLM_MODEL", "llama-3.1-8b")
    
    if _llm_handler is None:
        _llm_handler = FreeLLMHandler(
            backend=backend,
            model_name=model_name,
            **kwargs
        )
    
    return _llm_handler


def reset_llm_handler():
    """Reset global LLM handler instance."""
    global _llm_handler
    _llm_handler = None

