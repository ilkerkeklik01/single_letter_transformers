# Hafta 4, Gün 1: Embedding Modelleri ve Anlamsal Arama (Semantic Search)

Bu doküman, Magibu Akademi Uygulamalı Yapay Zekâ Mimarisi Eğitimi'nin 4. haftasının 1. gününde işlenen tüm teknik konuları, vektör uzaylarını ve anlamsal arama temellerini içermektedir.

---

## 📺 İlgili Video Kaynakları
- 🎥 [Hafta 4, Gün 1: Embedding Modelleri ve Anlamsal Arama](https://www.youtube.com/watch?v=LNkSTDezw8g)

---

## 1. Token Embedding vs Sentence / Document Embedding

LLM'lerdeki **Token Embedding**, her bir token'ın sözlükteki vektör temsilidir. **Sentence / Document Embedding** ise tüm bir cümlenin, paragrafın veya dokümanın anlamını temsil eden **tek bir sabit boyutlu vektördür** (Örn: 768, 1024, 3072 boyutlu).

### Anlam Uzayı (Vector Space):
Embedding modelleri metinleri anlam boyutlarına göre çok boyutlu bir küresel uzaya yerleştirir. Anlamca birbirine yakın cümleler (Örn: `"Türkiye'nin enflasyon trendi nedir?"` ile `"Son aylarda Türkiye'de enflasyon oranları nasıl değişti?"`) bu uzayda birbirine yakın noktalara düşer (%90+ kosinüs benzerliği).

---

## 2. Pooling (Havuzlama) Stratejileri

Encoder modelleri (BERT, Gemma Encoder, Qwen Encoder) dizideki her token için bir vektör üretir ($N 	imes D$). Tüm diziyi tek bir vektöre ($1 	imes D$) dönüştürmek için havuzlama yapılır:

1. **Mean Pooling (Ortalama Havuzlama):** Tüm token vektörlerinin ortalamasını alır. En yaygın yöntemdir.
2. **EOS / CLS Pooling:** Dizinin sonundaki `<eos>` veya başındaki `<cls>` token'ının vektörünü temsilci olarak seçer (Örn: Qwen3 Embedding).
3. **Max Pooling:** Her boyuttaki en yüksek değeri seçer.

---

## 3. Bi-Encoder vs Cross-Encoder Mimarileri

- **Bi-Encoder (Çift Kodlayıcı - Sentence Transformers):** Sorguyu ve Dokümanı ayrı ayrı vektöre dönüştürür. Vektör veritabanlarında milisaniyeler içinde milyonlarca dokümanda vektör araması (*Vector Search*) yapmak için kullanılır.
- **Cross-Encoder (Çapraz Kodlayıcı / Reranker):** Sorgu ve Dokümanı yan yana tek bir modele verir (`sorgu + doküman`). Tüm ikili etkileşimi hesaplar. Çok daha hassas skor verir ancak arama hızı yavaştır. Arama sonrası ilk $k$ sonucu yeniden sıralamak (*Reranking*) için kullanılır.

---

## 4. Matryoshka Embedding (MRL)

Matryoshka Representation Learning (MRL), vektörün ilk $N$ boyutunun (örneğin 768 boyuttan ilk 128 boyutunun) da anlamın büyük kısmını temsil etmesini sağlar. Veritabanı ve bellek maliyetlerini düşürmek için idealdir.

---

## 5. STS (Semantic Textual Similarity) Veri Kümesi ve Ödev

İki cümlenin anlamsal benzerliğini 0 ile 5 (veya 0 ile 10) arasında puanlayan etiketli veri kümesi hazırlama.

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("Magibu/embedding-magibu-200m")

sentences = [
    "Türkiye'de enflasyon oranları son aylarda nasıl değişti?",
    "Son aylarda Türkiye'deki enflasyon trendi nedir?",
    "Güneşli bir günde parka gitmek çok eğlencelidir."
]

embeddings = model.encode(sentences, convert_to_tensor=True)

# Kosinüs Benzerliği Hesaplama
sim_12 = util.cos_sim(embeddings[0], embeddings[1])
sim_13 = util.cos_sim(embeddings[0], embeddings[2])

print(f"Cümle 1 & 2 Benzerliği: {sim_12.item():.4f}") # ~0.90
print(f"Cümle 1 & 3 Benzerliği: {sim_13.item():.4f}") # ~0.14
```
