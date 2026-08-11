"""Embedding modellerini olcer ve karsilastirir.

Bir retriever'in iyi olup olmadigini "hissederek" anlayamazsiniz; olcmeniz gerekir.
Bu betik iki basit soru sorar:

    1. ISABET: KB icindeki bir soruda DOGRU makaleyi ilk sirada getiriyor mu?
    2. AYRIM : KB icindeki sorularin en dusuk skoru, KB disindaki sorularin
               en yuksek skorundan buyuk mu?

(2) negatifse, hicbir esik degeri ise yaramaz: ya alakali sorulari reddedersiniz
ya da alakasiz sorulara cevap uydurursunuz. Iyi bir retriever'da bu sayi pozitiftir.

Kullanim:
    python3 olcum_karsilastirma.py                    # indekslenmis tum modeller
    python3 olcum_karsilastirma.py --model gemma bge
"""

import argparse

import medical_rag
import ollama_client

# KB icinde olmasi gereken sorular ve beklenen makale basligindan bir parca.
IN_KB = [
    ("Bazofil nedir?", "Bazofil"),
    ("Ablasyon tedavisi nasil yapilir?", "Ablasyon"),
    ("Hemoglobin dusuklugu neden olur?", "Hemoglobin"),
    ("Adenit belirtileri nelerdir?", "Adenit"),
    ("Endoskopik sleeve gastroplasti nedir?", "Gastroplasti"),
    ("Kol agrisi neden olur?", "Kol Ağrısı"),
]

# KB'de kesinlikle olmayan sorular: asistan bunlara "Bilmiyorum" demeli.
OUT_OF_KB = [
    "Mars kolonilerinde grip nasil tedavi edilir?",
    "Bitcoin fiyati ne kadar?",
    "Kuantum bilgisayar nedir?",
    "Besiktas kac kupa kazandi?",
    "Bugun hava nasil?",
]

parser = argparse.ArgumentParser(description="Embedding modellerini karsilastir.")
parser.add_argument(
    "--model",
    nargs="+",
    default=list(ollama_client.EMBED_MODELS),
    choices=list(ollama_client.EMBED_MODELS),
    help="Olculecek modeller",
)
args = parser.parse_args()

for embed_key in args.model:
    config = ollama_client.EMBED_MODELS[embed_key]
    collection = medical_rag.get_collection(embed_key)
    print(f"\n=== {embed_key}  ({config['name']}) ===")
    if collection.count() == 0:
        print("  Bu model icin indeks bos. Once: "
              f"python3 index_medical.py --model {embed_key}")
        continue

    hits = 0
    in_scores = []
    for question, expected in IN_KB:
        top = medical_rag.search(question, embed_key, k=1)[0]
        correct = expected.lower() in top["title"].lower()
        hits += correct
        in_scores.append(top["similarity"])
        print(f"  {'OK ' if correct else 'YANLIS'} {top['similarity']:.3f}  "
              f"{question[:38]:40} -> {top['title'][:36]}")

    out_scores = [medical_rag.search(q, embed_key, k=1)[0]["similarity"] for q in OUT_OF_KB]

    gap = min(in_scores) - max(out_scores)
    print(f"  ---")
    print(f"  Isabet@1        : {hits}/{len(IN_KB)}")
    print(f"  KB-ici  en dusuk: {min(in_scores):.3f}  (ortalama {sum(in_scores)/len(in_scores):.3f})")
    print(f"  KB-disi en yuks.: {max(out_scores):.3f}")
    print(f"  AYRIM           : {gap:+.3f}  ->  {'kullanilabilir' if gap > 0 else 'KULLANILAMAZ'}")
    if gap > 0:
        print(f"  Onerilen esik   : {min(in_scores) - gap / 2:.2f}  "
              f"(su anki: {config['min_similarity']})")
