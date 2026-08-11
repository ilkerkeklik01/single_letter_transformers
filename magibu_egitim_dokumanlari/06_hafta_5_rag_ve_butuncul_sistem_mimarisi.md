# Hafta 5: RAG, Agentic AI ve Bütüncül Sistem Mimarisi (Bitirme Projesi Hazırlık Rehberi)

Bu doküman, **Magibu Akademi Uygulamalı Yapay Zekâ Mimarisi Eğitimi**'nin 5. haftasına ait tüm ders anlatımlarını, uçtan uca RAG ve Agentic AI mimari tasarımlarını, değerlendirme metriklerini ve **Bitirme Projesi Kılavuzu**'nu içermektedir.

---

## 📌 5. Hafta Genel Bakış ve Hedefler

Eğitimin ilk 4 haftası boyunca sırasıyla **Tokenizer Mimarileri**, **Model Özelleştirme (LoRA/QLoRA)**, **Ölçülebilirlik (Benchmarks)**, **Chat Templates**, **Structured Output / Tool Calling (MCP)**, **Embedding Modelleri** ve **Chunking / Vektör Veritabanları** konularını inceledik.

1. Hafta, tüm bu bağımsız yapay zekâ bileşenlerini birleştirerek **uçtan uca, otonom, araç kullanabilen ve doğrulanabilir bilgi getiren bütüncül bir yapay zekâ sistemi (Agentic RAG System)** inşa etme haftasıdır.

---

## 1. Gün 1: Uçtan Uca RAG Altyapısı ve Değerlendirme (Evaluation)

RAG (Retrieval-Augmented Generation), yalnızca LLM'e doküman vermekten ibaret değildir; iki ayrı zamanda çalışan 8 adımlı bir mühendislik hattıdır.

### 1.1 RAG İki Ana Akış

```text
İNDEKSLEME ZAMANI (Belge Değiştikçe)              SORGU ZAMANI (Kullanıcı Sordukça)
──────────────────────────────────              ─────────────────────────────────
[Dokümanlar]                                    [Kullanıcı Sorgusu]
     ↓                                                  ↓
[Metin Temizleme & Metadata]                    [Sorgu Embedding'i]
     ↓                                                  ↓
[Chunking (Parçalama)]                          [Vektör Arama & Metadata Filtresi]
     ↓                                                  ↓
[Chunk Embedding'leri]                          [Reranking (Çapraz Kodlayıcı)]
     ↓                                                  ↓
[Vektör Veritabanı Saklama]                     [Grounded Prompt İnşası]
                                                        ↓
                                                [LLM Yanıt Üretimi]
                                                        ↓
                                                [Cevap + Kaynak Gösterme]
```

### 1.2 Grounded Prompting ve Güvenlik (Prompt Injection)

Modelin kendi bilgisinden uydurmasını engellemek için sistem komutunda **Grounded Prompt** kullanılır:

```text
Sen yalnızca sana sağlanan BAĞLAM dokümanlarına dayanarak yanıt veren bir asistansın.
Sağlanan bağlamda kullanıcının sorusunun yanıtı YOKSA kesinlikle bilgi uydurma.
Doğrudan şu yanıtı ver: "Aradığınız bilgi sunulan dokümanlarda bulunmamaktadır."

BAĞLAM:
{retrieved_chunks}

SORU:
{user_query}
```

---

### 1.3 RAG Değerlendirme Metrikleri (RAG Evaluation)

Retriever (Getirici) ve Generator (Üretici) hatalarını birbirinden ayırmak için şu metrikler kullanılır:

1. **`Recall@k`:** Doğru/ilgili doküman parçası ilk $k$ aday sonuç arasında var mı? (0 veya 1).
2. **`MRR (Mean Reciprocal Rank)`:** İlk doğru dokümanın kaçıncı sırada geldiğinin tersinin ortalaması ($\frac{1}{ ext{rank}}$). Üst sıraları ödüllendirir.
3. **`Faithfulness / Adherence (Sadakat)`:** Üretilen yanıt, getirilen bağlam dokümanları tarafından destekleniyor mu? (Halüsinasyon tespiti).
4. **`Answer Correctness`:** Üretilen yanıt altın standart (*gold answer*) cevaba göre ne kadar doğru?
5. **`Abstention / Rejection Rate (Reddetme Oranı)`:** Bilgi dokümanda yokken sistem gerçekten *"Bulunamadı"* diyebiliyor mu?

---

### 1.4 Python ile Uçtan Uca RAG Kodu

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class SimpleRAGPipeline:
    def __init__(self, model_name="all-MiniLM-L6-v2", threshold=0.55):
        self.embedder = SentenceTransformer(model_name)
        self.threshold = threshold
        self.chunks = []
        self.vectors = None

    def index_documents(self, docs):
        """Dokümanları yükler ve L2-normalized vektörlerini saklar."""
        self.chunks = docs
        self.vectors = self.embedder.encode(docs, normalize_embeddings=True)

    def query(self, user_query):
        """Sorguyu vektörleştirir ve eşik değerine göre getirir."""
        query_vec = self.embedder.encode([user_query], normalize_embeddings=True)[0]
        similarities = np.dot(self.vectors, query_vec)
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score < self.threshold:
            return "Aradığınız sorunun cevabı dokümanlarımızda bulunmamaktadır.", []
        
        return self.chunks[best_idx], [best_idx]

# Kullanım Örneği
rag = SimpleRAGPipeline()
rag.index_documents([
    "Acil servis başvurularında triyaj yeşil, sarı ve kırmızı kod olarak üçe ayrılır.",
    "Poliklinik muayene randevuları MHRS sistemi üzerinden alınabilir."
])

context, sources = rag.query("Triyaj kodları nelerdir?")
print(f"Getirilen Bağlam: {context}")
```

---

## 2. Gün 2: Bütüncül Yapay Zekâ Sistemi Tasarımı (Agentic RAG)

Gelişmiş yapay zekâ sistemlerinde kullanıcı sorusu tek bir sabit akışa sokulmaz. Bir **Sorgu Yönlendirici (Intent Router)** soru türünü analiz ederek doğru aracı tetikler.

### 2.1 Mimarinin Yönlendirme Mantığı (Routing Logic)

```text
                             [Kullanıcı Girdisi]
                                      ↓
                           [Sorgu Yönlendirici / Intent]
                                      ↓
       ┌──────────────────┬───────────┴───────────┬──────────────────┐
       ↓                  ↓                       ↓                  ↓
 [RAG Arama]        [Web Arama]            [Tool / MCP]       [Serbest Sohbet]
(Doküman Verisi) (Güncel Dünya Bilgisi) (Sistem Aksiyonu/Kod)   (Genel LLM)
       └──────────────────┴───────────┬───────────┴──────────────────┘
                                      ↓
                          [LLM Bütüncül Yanıt Üretimi]
                                      ↓
                          [Kullanıcıya Sunum & UI]
```

---

## 3. Bitirme Projesi (Graduation Capstone Project) Kılavuzu

### Proje Konusu ve Şartları

Katılımcılar seçtikleri özel bir domain (Örn: Tıp, Hukuk, Finans, E-Ticaret, Eğitim, Müşteri Hizmetleri vb.) için **Model + Tool/MCP + RAG/Vektör Veritabanı** içeren canlı bir yapay zekâ ajanı geliştirecektir.

### Zorunlu Bileşenler

1. **RAG / Vektör Veritabanı Entegrasyonu:** Seçilen alana ait dokümanların chunking ve embedding işlemlerinden geçirilerek sorgulanması.
2. **Tool Calling / MCP Kullanımı:** Dış veritabanı, hesaplama veya API çağrısı yapabilen en az 2 fonksiyon.
3. **Structured Output & Chat Template:** Standart Jinja2 sohbet şablonu ve JSON kısıtlı örnekleme kullanımı.
4. **Hugging Face Space Üzerinde Canlı Uygulama:** Gradio veya Streamlit ile hazırlanmış, yapılan arka plan işlemlerini (*Tool Call*, *Retrieved Chunk*, *Score*) gösteren kullanıcı dostu arayüz.

---

## 🏅 Doğrulanabilir Başarı Sertifikası Şartları

Süreci başarıyla tamamlayıp resmi **Magibu Akademi Başarı Sertifikası** almaya hak kazanmak için aşağıdaki koşulların eksiksiz sağlanması gerekir:

1. **GitHub ve Hugging Face Repository Teslimleri:**
   - Projenin açık kaynak GitHub kodu ve Hugging Face Space canlı bağlantısı.
2. **Kapsamlı `README.md` Dokümantasyonu:**
   - Sistem mimarisi şeması.
   - Kullanılan embedding modeli ve chunking stratejisinin gerekçeleri.
   - RAG Değerlendirme (`Recall@k`, ret oranı) ve Benchmark sonuçları.
3. **Canlı Uygulama Doğrulaması:**
   - Hugging Face Space üzerinde uygulamanın donmadan, hatalara karşı dirençli biçimde çalışması.
   - Cevabı dokümanda olmayan sorularda sistemin halüsinasyon görmeden ret yanıtı vermesi.
