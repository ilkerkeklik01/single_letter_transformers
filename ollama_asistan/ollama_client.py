"""Ollama HTTP API'si icin ince bir sarmalayici.

Ollama, bilgisayarinizda 11434 portunda calisan basit bir HTTP sunucusudur.
Ekstra bir kutuphaneye ihtiyacimiz yok: iki tane POST istegi her seyi yapar.

    POST /api/embed   -> metni vektore cevirir
    POST /api/chat    -> sohbet eder, gerektiginde arac cagirir
"""

import os

import requests

BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Arac cagirabilen (tool calling) sohbet modeli.
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "muse-glimmer:30b-mlx")

# Iki farkli embedding modeli: birini secip karsilastirabilirsiniz.
#
# "onek" (prefix) meselesi: bazi embedding modelleri, metnin BELGE mi yoksa SORU mu
# oldugunu belirten kisa bir on ek ile egitilir. Bu oneki koymazsaniz model calisir
# ama isabet belirgin sekilde duser. EmbeddingGemma tam olarak boyle bir modeldir.
# "min_similarity" ise o modele ozel alaka esigidir. Neden model basina degisir?
# Cunku her modelin benzerlik skorlari farkli bir araliga yayilir; birinde 0.50
# "alakasiz" demekken digerinde "cok alakali" olabilir. Esigi tek bir sabit
# olarak yazmak, model degistirdiginizde sessizce bozulur. Bu degerler
# olcum_karsilastirma.py ile olculerek secilmistir.
EMBED_MODELS = {
    "gemma": {
        "name": "embeddinggemma:latest",  # Google, cok dilli, 768 boyut
        "query_prefix": "task: search result | query: ",
        "doc_prefix": "title: none | text: ",
        "min_similarity": 0.48,
    },
    "bge": {
        "name": "bge-m3:latest",  # BAAI, cok dilli, 1024 boyut
        "query_prefix": "",
        "doc_prefix": "",
        "min_similarity": 0.55,
    },
    # UYARI: Bu model bu gorevde iyi ayrim yapamiyor (olculen degerler README'de).
    # Egitimde "kotu retriever nasil anlasilir?" ornegi olarak dursun diye birakildi.
    "magibu": {
        "name": "alibayram/embeddingmagibu-200m:latest",  # Turkce'ye ozel, 768 boyut
        "query_prefix": "",
        "doc_prefix": "",
        "min_similarity": 0.75,
    },
}
DEFAULT_EMBED = "gemma"

CONNECTION_ERROR = (
    f"Ollama'ya baglanilamadi ({BASE_URL}). "
    "Once 'ollama serve' komutunu calistirin ya da Ollama uygulamasini acin."
)


def _post(path: str, payload: dict, timeout: int = 600) -> dict:
    """Ollama'ya POST atar, JSON cevabi dondurur."""
    try:
        response = requests.post(f"{BASE_URL}{path}", json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(CONNECTION_ERROR) from exc
    if response.status_code != 200:
        raise RuntimeError(f"Ollama hatasi ({response.status_code}): {response.text[:300]}")
    return response.json()


def embed(texts: list[str], embed_key: str = DEFAULT_EMBED, kind: str = "doc") -> list[list[float]]:
    """Metin listesini vektor listesine cevirir.

    embed_key: EMBED_MODELS icindeki anahtar ("gemma" ya da "magibu").
    kind:      "doc" (indekslenen makale parcasi) ya da "query" (kullanici sorusu).

    Ayni metni "doc" ve "query" olarak gomerseniz farkli vektorler alirsiniz;
    bu bir hata degil, modelin egitildigi bicimdir.
    """
    if embed_key not in EMBED_MODELS:
        raise ValueError(f"Bilinmeyen embedding modeli: {embed_key}. Secenekler: {list(EMBED_MODELS)}")
    if kind not in {"doc", "query"}:
        raise ValueError(f"kind 'doc' ya da 'query' olmali, '{kind}' verildi.")

    config = EMBED_MODELS[embed_key]
    prefix = config["query_prefix"] if kind == "query" else config["doc_prefix"]
    data = _post("/api/embed", {"model": config["name"], "input": [prefix + t for t in texts]})
    return data["embeddings"]


def chat(
    messages: list[dict],
    model: str = CHAT_MODEL,
    tools: list[dict] | None = None,
    temperature: float = 0.1,
) -> dict:
    """Sohbet mesajlarini modele gonderir ve model mesajini dondurur.

    Donen sozlukte "content" (metin) ve varsa "tool_calls" (arac cagrilari) bulunur.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,   # tek parca cevap; ogrenmesi daha kolay
        "think": False,    # dusunme blogunu kapat, ciktiyi sade tut
        "options": {"temperature": temperature},
    }
    if tools:
        payload["tools"] = tools
    return _post("/api/chat", payload)["message"]
