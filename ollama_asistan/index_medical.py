"""Ilk 100 tibbi makaleyi indirir, parcalar, vektorler ve ChromaDB'ye yazar.

Kullanim:
    python3 index_medical.py                        # 100 makale, embeddinggemma
    python3 index_medical.py --model both --reset   # iki modeli de bastan indeksle
"""

import argparse
import time

import chromadb
from datasets import load_dataset

import medical_rag
import ollama_client

DATASET = "umutertugrul/turkish-hospital-medical-articles"
BATCH_SIZE = 32

parser = argparse.ArgumentParser(description="Tibbi makaleleri vektor veritabanina yaz.")
parser.add_argument("--limit", type=int, default=100, help="Kac makale indekslensin (varsayilan 100)")
parser.add_argument("--split", default="acibadem", help="Hastane bolumu (varsayilan acibadem)")
parser.add_argument(
    "--model",
    default="gemma",
    choices=[*ollama_client.EMBED_MODELS, "both"],
    help="Hangi embedding modeli kullanilsin ('both' = gemma + bge)",
)
parser.add_argument("--reset", action="store_true", help="Koleksiyonu sifirdan olustur")
args = parser.parse_args()

print(f"[1/3] Veri seti yukleniyor: {DATASET} (bolum: {args.split})")
dataset = load_dataset(DATASET, split=args.split).select(range(args.limit))
print(f"      {len(dataset)} makale alindi.")

print("[2/3] Makaleler parcalara bolunuyor...")
documents: list[str] = []
metadatas: list[dict] = []
ids: list[str] = []
for article_index, row in enumerate(dataset):
    for chunk_index, chunk in enumerate(medical_rag.chunk_text(row["text"])):
        documents.append(chunk)
        metadatas.append(
            {"title": row["title"], "url": row["url"], "article_index": article_index}
        )
        ids.append(f"{article_index}-{chunk_index}")
print(f"      {len(documents)} parca olustu (makale basina ~{len(documents) // len(dataset)}).")

embed_keys = ["gemma", "bge"] if args.model == "both" else [args.model]
client = chromadb.PersistentClient(path=medical_rag.DB_PATH)

for embed_key in embed_keys:
    name = f"tibbi_makaleler_{embed_key}"
    if args.reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass  # koleksiyon zaten yoksa sorun degil

    collection = medical_rag.get_collection(embed_key)
    print(f"[3/3] '{ollama_client.EMBED_MODELS[embed_key]['name']}' ile vektorler uretiliyor...")
    started = time.time()

    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        collection.add(
            ids=ids[start : start + BATCH_SIZE],
            documents=batch,
            metadatas=metadatas[start : start + BATCH_SIZE],
            embeddings=ollama_client.embed(batch, embed_key, kind="doc"),
        )
        done = min(start + BATCH_SIZE, len(documents))
        print(f"      {done}/{len(documents)} parca ({time.time() - started:.0f} sn)", end="\r")

    print(f"\n      Bitti: '{name}' koleksiyonunda {collection.count()} parca "
          f"({time.time() - started:.0f} sn).")

print("\nHazir. Simdi sohbeti baslatin:  python3 chat.py")
