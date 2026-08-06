"""Retrieved bağlamdan cevap üreten extractive, Ollama ve Transformers backend'leri."""

import re
import unicodedata
from typing import Protocol

from .embeddings import _post_json


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


class ExtractiveGenerator:
    """LLM değildir; çevrimdışı demoda prompt içinden en uygun cümleyi seçer."""

    def generate(self, prompt: str) -> str:
        question_match = re.search(r"SORU:\n(.+?)\n\n", prompt, flags=re.DOTALL)
        context_match = re.search(r"BAĞLAM:\n(.+?)\n\nCEVAP:", prompt, flags=re.DOTALL)
        if not question_match or not context_match:
            return "Bağlamda bu sorunun cevabı bulunamadı."
        question_terms = _terms(question_match.group(1))
        sentences = re.split(r"(?<=[.!?])\s+", context_match.group(1))
        ranked = sorted(
            ((_lexical_score(question_terms, _terms(sentence)), sentence) for sentence in sentences),
            reverse=True,
        )
        if not ranked or ranked[0][0] == 0:
            return "Bağlamda bu sorunun cevabı bulunamadı."
        return ranked[0][1].strip()


class OllamaGenerator:
    def __init__(
        self,
        model_name: str = "gemma3",
        base_url: str = "http://localhost:11434",
        timeout: float = 180.0,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        response = _post_json(f"{self.base_url}/api/generate", payload, self.timeout)
        return str(response.get("response", "")).strip()


class TransformersGenerator:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_new_tokens: int = 180,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers backend için: pip install -r requirements-optional.txt"
            ) from exc
        self._torch = torch
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    def generate(self, prompt: str) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _terms(text: str) -> set[str]:
    folded = unicodedata.normalize("NFC", text).casefold().replace("i\u0307", "i")
    return set(re.findall(r"\w+", folded, flags=re.UNICODE))


def _lexical_score(question_terms: set[str], sentence_terms: set[str]) -> int:
    """Tam eşleşmeye 2, en az dört harflik ortak öneke 1 puan verir."""
    score = 0
    for query_term in question_terms:
        if query_term in sentence_terms:
            score += 2
        elif len(query_term) >= 4 and any(
            len(term) >= 4
            and (query_term.startswith(term[:4]) or term.startswith(query_term[:4]))
            for term in sentence_terms
        ):
            score += 1
    return score
