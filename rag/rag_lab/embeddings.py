"""Aynı ``embed(list[str])`` sözleşmesini kullanan üç embedding backend'i."""

import hashlib
import json
import re
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Protocol

from .similarity import normalize


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashEmbedding:
    """İndirme gerektirmeyen demo modeli; semantik model değildir.

    Kelime ve karakter n-gram'larını sabit boyutlu bir hashing trick uzayına
    taşır. Pipeline'ı çevrimdışı öğretmek ve test etmek içindir.
    """

    def __init__(self, dimension: int = 256):
        if dimension <= 0:
            raise ValueError("dimension pozitif olmalı")
        self.dimension = dimension
        self.model_name = f"hash-demo-{dimension}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        normalized_text = _casefold_turkish(text)
        words = re.findall(r"\w+", normalized_text, flags=re.UNICODE)
        features = [f"w:{word}" for word in words]
        compact = " ".join(words)
        features.extend(f"c3:{compact[i:i + 3]}" for i in range(max(0, len(compact) - 2)))
        vector = [0.0] * self.dimension
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=9).digest()
            index = int.from_bytes(digest[:8], "little") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        if not any(vector):
            vector[0] = 1.0
        return normalize(vector)


class OllamaEmbedding:
    """Ollama'nın yerel ``POST /api/embed`` endpoint'ini kullanır."""

    def __init__(
        self,
        model_name: str = "embeddinggemma",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model_name, "input": list(texts), "truncate": False}
        response = _post_json(f"{self.base_url}/api/embed", payload, self.timeout)
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama beklenen sayıda embedding döndürmedi")
        return [[float(value) for value in row] for row in embeddings]


class TransformersEmbedding:
    """AutoModel gizli durumlarına maskeli mean pooling uygular.

    Her ``AutoModel`` iyi bir retrieval modeli değildir. Varsayılan model cümle
    benzerliği için eğitilmiştir; model kartındaki query/document talimatlarını
    ayrıca kontrol etmek gerekir.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
        max_length: int = 512,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers backend için: pip install -r requirements-optional.txt"
            ) from exc
        self._torch = torch
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        torch = self._torch
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().float().tolist()


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"{url} çağrısı başarısız. Servis/model hazır mı? Ayrıntı: {exc}"
        ) from exc


def _casefold_turkish(text: str) -> str:
    """Unicode casefold'un ``İ -> i + combining dot`` sonucunu arama için birleştirir."""
    folded = unicodedata.normalize("NFC", text).casefold().replace("i\u0307", "i")
    return unicodedata.normalize("NFC", folded)
