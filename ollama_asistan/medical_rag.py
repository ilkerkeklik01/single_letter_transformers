"""Tibbi makaleler uzerinde RAG: parcalama, vektor arama ve TOPRAKLANMIS cevap.

Bu dosyanin tek bir amaci var: model, sadece bulunan makalelerde yazani soylesin.
Bunu iki kapi ile saglariz:

    1. ARAMA KAPISI     -> benzerlik esigin altindaysa LLM hic cagrilmaz.
    2. URETIM KAPISI    -> LLM'e "sadece bu metinlerden cevapla" talimati verilir.

Ikisinden biri gecilemezse cevap: REFUSAL ("Bilmiyorum ...").
"""

import os

import chromadb

import ollama_client

# Vektorler bu klasorde diske yazilir; silmek indeksi sifirlar.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# Bilgi bulunamadiginda verilecek sabit cevap.
REFUSAL = "Bilmiyorum — bu bilgi mevcut tibbi makalelerde bulunmuyor."

SYSTEM_PROMPT = f"""Sen bir tibbi bilgi asistanisin.

Kurallar:
1. SADECE asagida verilen makale parcalarindaki bilgiyi kullan.
2. Genel tip bilgini KESINLIKLE kullanma. Metinde yoksa, senin icin yoktur.
3. Parcalar soruyu cevaplamaya yetmiyorsa, aynen su cumleyi yaz ve baska hicbir sey ekleme:
{REFUSAL}
4. Cevabini Turkce, kisa ve sade yaz. Tani koyma, ilac dozu onerme.
"""


def chunk_text(text: str, size: int = 220, overlap: int = 40) -> list[str]:
    """Uzun metni, ucları birbirine binen kelime pencerelerine boler.

    Bindirme (overlap) sayesinde bir cumle tam iki parcanin arasina dusup kaybolmaz.
    """
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if window:
            chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def get_collection(embed_key: str = ollama_client.DEFAULT_EMBED):
    """Secilen embedding modeline ait Chroma koleksiyonunu acar/olusturur.

    Her model icin AYRI koleksiyon kullaniriz: iki modelin vektorleri
    farkli uzaylarda yasar, karistirilirsa arama sonuclari anlamsiz olur.
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        name=f"tibbi_makaleler_{embed_key}",
        metadata={"hnsw:space": "cosine"},  # mesafe = 1 - kosinus benzerligi
        embedding_function=None,            # vektorleri biz uretiyoruz (Ollama ile)
    )


def search(question: str, embed_key: str = ollama_client.DEFAULT_EMBED, k: int = 4) -> list[dict]:
    """Soruyu vektore cevirip en yakin k parcayi getirir."""
    collection = get_collection(embed_key)
    if collection.count() == 0:
        return []

    query_vector = ollama_client.embed([question], embed_key, kind="query")[0]
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "text": text,
                "title": meta.get("title", "-"),
                "url": meta.get("url", "-"),
                "similarity": 1.0 - distance,
            }
        )
    return hits


def answer_medical(question: str, embed_key: str = ollama_client.DEFAULT_EMBED, k: int = 4) -> dict:
    """Tibbi soruyu sadece indekslenmis makalelere dayanarak cevaplar."""
    hits = search(question, embed_key, k)

    # --- 1. KAPI: arama ---
    # Hicbir parca esigi gecemediyse LLM'i hic cagirmayiz; uyduramaz.
    # Esik, kullanilan embedding modeline gore degisir (bkz. ollama_client.EMBED_MODELS).
    threshold = ollama_client.EMBED_MODELS[embed_key]["min_similarity"]
    relevant = [h for h in hits if h["similarity"] >= threshold]
    if not relevant:
        return {"answer": REFUSAL, "sources": [], "grounded": False}

    # --- 2. KAPI: uretim ---
    context = "\n\n".join(
        f"[{i}] {h['title']}\n{h['text']}" for i, h in enumerate(relevant, start=1)
    )
    message = ollama_client.chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"MAKALE PARCALARI:\n{context}\n\nSORU: {question}"},
        ],
        temperature=0.0,  # yaraticilik yok, sadece metne sadakat
    )
    answer = (message.get("content") or "").strip() or REFUSAL

    # Model "Bilmiyorum" dediyse kaynak gostermenin anlami yok.
    grounded = REFUSAL.split("—")[0].strip().lower() not in answer.lower()

    # Ayni makaleden birden fazla parca gelmis olabilir; kaynak listesinde
    # her makale bir kez, en yuksek benzerlik skoruyla gorunsun.
    sources: list[dict] = []
    if grounded:
        seen: set[str] = set()
        for hit in relevant:  # relevant zaten benzerlige gore sirali
            if hit["url"] not in seen:
                seen.add(hit["url"])
                sources.append(
                    {
                        "title": hit["title"],
                        "url": hit["url"],
                        "similarity": round(hit["similarity"], 3),
                    }
                )

    return {"answer": answer, "sources": sources, "grounded": grounded}
