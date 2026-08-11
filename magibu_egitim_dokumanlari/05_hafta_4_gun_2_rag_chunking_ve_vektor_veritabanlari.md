# Magibu Uygulamalı Yapay Zekâ Mimarisi | Hafta 4, Gün 2: RAG, Chunking ve Vektör Veritabanları

Bu doküman, **Magibu Akademi Uygulamalı Yapay Zekâ Mimarisi Eğitimi** kapsamında işlenen **Hafta 4, Gün 2** dersinin tüm teorik, matematiksel, mühendislik ve uygulama detaylarını içermektedir.

---

## 📺 İlgili Video Kaynakları
- 🎥 [Magibu Uygulamalı Yapay Zekâ Mimarisi | Hafta 4, Gün 2: RAG, Chunking ve Vektör Veritabanları](https://www.youtube.com/watch?v=X8hvbvCigzc)

---

## 1. Dersin Temel Amacı ve Giriş

Bu dersin temel amacı, yapay zekâ sistemlerinde **RAG (Retrieval-Augmented Generation)** altyapısının teorik ve pratik temellerini kurmak; metinlerin anlamsal uzaya taşınması, parçalanması (**chunking**) ve vektör veritabanlarında saklanıp sorgulanması süreçlerini mühendislik disipliniyle incelemektir.

### Önemli Notlar:
- LLM'ler tek başlarına yalnızca sıradaki token'ı tahmin eden (*next-token generation*) yapılardır. Şirket verileri, tıbbi dokümanlar veya güncel bilgiler sorulduğunda doğrudan LLM'e bırakılırsa **halüsinasyon (hallucination)** riski doğar.
- RAG, modele bilgi uydurtmak yerine elimizdeki güvenilir bilgi kaynaklarından doğru parçaları bulup LLM'e bağlam (context) olarak sunma sanatıdır. Hatta bazı senaryolarda hiç LLM kullanmadan yalnızca **Vektör Arama (Vector Search)** ile doğrudan doğru dokümanı kullanıcıya getirmek mümkündür.

---

## 2. Anlamsal Uzay, Vektörler ve Matematiksel Temeller

Metinler anlamsal uzaya taşındığında her bir metin veya doküman $N$ boyutlu uzayda (örneğin 768 veya 3072 boyutlu) tek bir **nokta (vektör)** olarak temsil edilir.

### 2.1 Mesafe Ölçümleri (Distance Metrics)

İki nokta veya metin arasındaki benzerliği ölçmek için kullanılan temel geometri yöntemleri:

1. **Öklid Mesafesi (Euclidean Distance):** İki nokta arasındaki düz çizgi (kuş bakışı) mesafesidir.
2. **Manhattan Mesafesi (City Block / Taksi Mesafesi):** Eksenler boyunca bina etrafından yürür gibi yapılan mesafe ölçümüdür.
3. **Kosinüs Benzerliği (Cosine Similarity):** Vektörlerin uzunluğundan ziyade **aralarındaki açıya ($	heta$)** bakar. Yapay zekâ ve embedding dünyasında en yaygın kullanılan yöntemdir.

$$	ext{similarity} = \cos(	heta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|_2 \|\mathbf{B}\|_2}$$

### 2.2 L2 Normalizasyonu ve Noktasal Çarpım (Dot Product)

Vektörler **L2 Normalizasyonu** işleminden geçirilerek uzunlukları $1$ birim olan birim çember / birim küre yüzeyine yapıştırılır. 

- Vektörler L2-normalized olduğunda $\|\mathbf{A}\|_2 = 1$ ve $\|\mathbf{B}\|_2 = 1$ olur.
- Bu durumda Kosinüs Benzerliği formülü doğrudan iki vektörün **Noktasal Çarpımına (Dot Product / $\mathbf{A} \cdot \mathbf{B}$)** dönüşür. Bu sayede milyonlarca vektör arasında matris çarpımı ($\mathbf{Q} \cdot \mathbf{D}^T$) ile çok hızlı arama yapılabilir.

### 2.3 Kosinüs Skorlarının Sezgisel Anlamı:
- **$+1$ (0° Açı):** İki metin anlamsal olarak tamamen aynı yöndedir.
- **$0$ (90° Açı / Dik Vektörler):** İki metin arasında hiçbir anlamsal ilişki veya gölge düşmesi yoktur.
- **$-1$ (180° Açı):** İki metin arasında tam zıt anlamsallık vardır.

---

## 3. Metin Parçalama (Chunking) Mühendisliği

8.000 veya 100.000 kelimelik koca bir makaleyi tek bir vektöre indirgersek, vektör dokümandaki tüm farklı konuların ortalamasını almaya başlar ve spesifik detaylar kaybolur. Bu nedenle dokümanlar **chunk (parça)** adı verilen mantıksal birimlere bölünür.

### Chunking Stratejileri:
1. **Sabit Boyutlu / Token Tabanlı Chunking (Fixed-size / Token Chunking):** Metin sabit $N$ token/kelimede bir bölünür. Anlam kopmasını önlemek için %10–20 oranında **Overlapping (üst üste bindirme)** uygulanır.
2. **Cümle Tabanlı Chunking (Sentence-level Chunking):** Metin noktalarla cümlelere ayrılır. SSS (FAQ) veya kısa yanıtlı sistemler için uygundur.
3. **Paragraf / Yapı-Duyarlı Chunking (Structure-aware / Markdown Chunking):** Paragraf sonları (`

`) ve Markdown başlıkları dikkate alınarak anlam bütünlüğü korunur.
4. **Semantik Chunking (Semantic Chunking):** Ardışık cümlelerin embedding vektörleri arasındaki kosinüs benzerliği hesaplanır. Benzerlik belirli bir eşik değerinin altına düştüğü yerde metin yeni bir chunk'a bölünür.
5. **Karma Chunking (Mixed Chunking):** Paragraf boyutuna göre dinamik kurallar uygulanır (Örn: 100 kelimeyi geçmeyen paragraflar tek chunk yapılır, geçenler kelime sınırıyla bölünür).
6. **Hiyerarşik Chunking (Child-to-Parent / Parent-Document Chunking):** Doküman küçük parçalara (child chunk) bölünerek arama başarısı (hit rate) artırılır; arama sonucunda ise LLM'e daha geniş bağlam sağlayan ana paragraf/doküman (parent) verilir.

---

## 4. Vektör Veritabanı Yapısı ve Mimari

Vektör veritabanı, üretilen embedding vektörlerini, orijinal metin parçalarını ve ilişkili metadata'yı saklayan ve hızlı en yakın komşu araması (ANN / KNN) yapan sistemdir.

### Asgari Vektör Veritabanı Şeması (3 Temel Sütun):
1. **`article_id` / `source_link`:** Parçanın ait olduğu orijinal makalenin veya dokümanın benzersiz kimliği.
2. **`chunk_text`:** Parçalanmış metnin kendisi.
3. **`embedding_vector`:** Metnin $N$ boyutlu sayısal vektörü (Float dizisi).
4. **`metadata` (İsteğe Bağlı):** Tarih, kategori, yazar, parent_id vb. ek bilgiler.

### Kullanılabilecek Veritabanları:
- **In-Memory / NumPy Exact Search:** Küçük veri kümeleri için en doğru ve sade yöntem.
- **SQLite Vector Store:** İlişkisel veritabanında kalıcı vektör saklama.
- **ChromaDB / PGVector / Qdrant / Weaviate:** Üretim ortamları için gelişmiş vektör arama motorları.

---

## 5. Hafta 4, Gün 2 Ödev Rehberi (Adım Adım)

### Ödevin Amacı:
Hugging Face üzerindeki tıp makalelerini chunking işleminden geçirip vektörleştirerek bir RAG veri kümesi oluşturmak ve eşik değerlerine (thresholding) göre sorgu yanıtlayan bir arama mekanizması kurmak.

---

### Adım 1: Veri Kümesini Seçme ve Hazırlama
Hugging Face üzerindeki **`Magibu`** organizasyonuna ait tıp makalelerinden **100 ile 1000 adet** makale seçin:
- [`Magibu/Turkish-Hospital-Medical-Articles`](https://huggingface.co/datasets/Magibu/Turkish-Hospital-Medical-Articles)
- [`Magibu/Turkish-Medical-Articles`](https://huggingface.co/datasets/Magibu/Turkish-Medical-Articles)

---

### Adım 2: Chunking ve Vektörleştirme
1. Seçtiğiniz makaleleri belirlediğiniz bir chunking stratejisiyle (örneğin 50-100 kelimelik paragraflar veya overlapping ile) parçalara bölün.
2. Bir embedding modeli (Örn: `SentenceTransformers`, Ollama `embeddinggemma` veya Hugging Face modelleri) kullanarak her chunk'ın vektörünü hesaplayın.
3. En az 3 sütundan oluşan veri kümenizi hazırlayın:
   - `article_id`
   - `chunk_text`
   - `embedding_vector`
4. Veri kümenizi Hugging Face Hub üzerinde bir repository olarak yayımlayın.

---

### Adım 3: Soru Kümesi Oluşturma (30 Soru)
Sisteminizin doğruluk ve ret yeteneğini test etmek için **30 adet soru** hazırlayın:
- **20 Adet Cevabı Makalelerde OLAN Soru:** Cevabının seçtiğiniz makalelerde yer aldığından emin olduğunuz spesifik sorular.
- **10 Adet Cevabı Makalelerde OLMAYAN Soru:** Veri kümesinde cevabı bulunmayan dış sorular (Sistemin uydurmasını / halüsinasyon görmesini engellemek için).

---

### Adım 4: Kosinüs Araması ve Eşik Değeri (Thresholding) Mantığı
Soruları vektörleştirip veritabanınızdaki chunk vektörleriyle kosinüs benzerliklerine göre karşılaştırın:
- **Yüksek Benzerlik (Örn: $> \%79$ veya $> \%96$):** İlgili makale/chunk doğrudan bulunur ve yanıtlanır.
- **Düşük Benzerlik (Örn: $< \%48$):** Veritabanında uygun cevap bulunamadığı tespit edilir ve sistem otomatik olarak:
  > *"Aradığınız sorunun cevabı dokümanlarımızda bulunmamaktadır."*
  mesajını döner. LLM'in uydurma yapmasına izin verilmez.

---

### Adım 5: Repository ve README Şablonu
Hugging Face repository'nizin `README.md` dosyasında şu başlıkları mutlaka açıklayın:
1. **Kullanılan Chunking Stratejisi:** Neden bu parça boyutunu ve yöntemi seçtiniz?
2. **Kullanılan Embedding Modeli:** Hangi model seçildi, kaç boyutlu vektör üretildi?
3. **Vektör Veritabanı / Arama Mantığı:** Arama nasıl yapıldı (NumPy, ChromaDB, PGVector vb.)?
4. **Eşik Değerleri (Thresholds) ve Sonuçlar:** Hangi benzerlik oranında ret cevabı verildi? 30 sorudaki başarı oranınız ne oldu?

---

## 🛠️ Örnek Python Kod Yapısı (Minimal RAG Search)

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Embedding Modelini Yükle
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Örnek Chunk'lar ve Vektörleştirme
chunks = [
    "Aktif akne tedavisi cilt tipine uygun temizleyiciler ve salisilik asit ile yapılır.",
    "Bkz. Kardiyoloji kliniğinde hipertansiyon takibi düzenli kan basıncı ölçümü gerektirir.",
    "Göz kuruluğu için suni gözyaşı damlaları doktor kontrolünde kullanılmalıdır."
]
chunk_vectors = model.encode(chunks, normalize_embeddings=True)

# 3. Sorgu Vektörü ve Kosinüs Araması
query = "Sivilcelerim için aktif akne tedavisini araştırıyorum, ne yapmalıyım?"
query_vector = model.encode([query], normalize_embeddings=True)[0]

# Normalized vektörlerde Dot Product = Cosine Similarity
similarities = np.dot(chunk_vectors, query_vector)

THRESHOLD = 0.55
best_idx = np.argmax(similarities)
best_score = similarities[best_idx]

print(f"En Yüksek Benzerlik Skoru: {best_score:.4f}")

if best_score >= THRESHOLD:
    print(f"Bulunan Bağlam: {chunks[best_idx]}")
else:
    print("Aradığınız sorunun cevabı dokümanlarımızda bulunmamaktadır.")
```
