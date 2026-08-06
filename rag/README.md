# RAG Temelleri — küçük parçalardan uçtan uca sisteme

Bu klasör RAG'i bir framework'ün tek satırlık çağrısı olarak değil, ayrı ayrı
ölçülebilen bileşenler olarak kurar. Örnekler Türkçe, küçük ve okunabilirdir.
İlk dört demo yalnızca Python standart kütüphanesiyle çalışır; Ollama ve
Transformers yolları aynı arayüzün gerçek model kullanan alternatifleridir.

> `HashEmbedding` ve `ExtractiveGenerator` gerçek üretim modelleri değildir.
> İnternet ve model indirmesi olmadan veri akışını görünür kılan öğretim
> araçlarıdır. Semantik kalite deneyi için Ollama veya Transformers örneğini
> kullanın.

## Hızlı başlangıç

Komutları bu klasörde çalıştırın:

```bash
cd rag

python3 -m examples.01_cosine
python3 -m examples.02_chunking
python3 -m examples.03_vector_store
python3 -m examples.04_rag_without_llm "İade ücretim ne zaman yansır?"
python3 -m examples.07_evaluate_retriever

python3 -m unittest discover -s tests -v
```

Ders sırasında hücre hücre ilerlemek için
[`rag_demo.ipynb`](rag_demo.ipynb) notebook'unu açın. Bağımlılıksız ana akış
cosine'dan değerlendirmeye kadar çalışır; Ollama ve Transformers hücreleri
varsayılan olarak kapalıdır.

## RAG'in iki ayrı zamanı

RAG, “LLM'e belge vermek”ten daha büyük bir sistemdir. İki ana akış vardır:

```text
INDEXLEME (belge değişince)                    SORGU (her kullanıcı sorusunda)
──────────────────────────                    ────────────────────────────────
belge                                          soru
  ↓                                              ↓
temizle / erişim metadata'sı ekle              query embedding
  ↓                                              ↓
chunking                                       top-k vektör arama + filtre
  ↓                                              ↓
document embedding                            gerekirse reranking
  ↓                                              ↓
vektör + metin + metadata sakla                bağlam + soru → prompt → LLM
                                                  ↓
                                                cevap + kaynaklar
```

Temel adımlar:

1. **Yükle:** PDF, HTML, Markdown veya veritabanı satırını metin ve metadata'ya
   dönüştür. Kaynak kimliği ve erişim yetkisini kaybetme.
2. **Parçala:** Model bağlamına ve bilgi yapısına uygun chunk'lar üret.
3. **Embed et:** Her parçayı aynı embedding modeli ve aynı sürümle vektöre çevir.
4. **İndeksle:** Vektör, chunk metni, belge kimliği ve metadata'yı birlikte sakla.
5. **Sorgula:** Soruyu, indekslemede kullanılan modelle vektöre çevir.
6. **Getir:** Benzerlik + metadata filtresiyle adayları bul; gerekirse rerank et.
7. **Üret:** Soruyu ve sınırlı bağlamı LLM'e ver; yetersiz bağlamda cevap vermemesini iste.
8. **Kaynak göster ve ölç:** Cevapla birlikte chunk kimliklerini döndür; retrieval
   ile generation hatalarını ayrı izle.

Bu ayrım hata ayıklamanın anahtarıdır. Doğru parça ilk `k` içinde yoksa sorun
generator'da değildir. Doğru parça var ama cevap yanlışsa embedding modelini
değiştirmek tek başına çözüm olmayabilir.

## 1. Cosine benzerliği

Embedding modeli metni bir vektöre dönüştürür. Cosine, iki vektörün uzunluğundan
çok aralarındaki açıya bakar:

```text
cos(a,b) = (a · b) / (||a||₂ ||b||₂)
```

| Vektörler | Sonuç | Sezgi |
| --- | ---: | --- |
| `[1,0]`, `[10,0]` | `+1` | uzunluk farklı, yön aynı |
| `[1,0]`, `[0,3]` | `0` | dik yönler |
| `[1,1]`, `[-1,-1]` | `-1` | zıt yönler |

Vektörler önce L2-normalize edilirse cosine yalnızca dot product olur. Bu yüzden
birçok motor normalize edilmiş matris üzerinde hızlıca `query @ documents.T`
hesaplar. Sıfır vektör için açı tanımsızdır; boyutlar da eşit olmalıdır.

Çalışan karşılıklar: [`rag_lab/similarity.py`](rag_lab/similarity.py) ve
[`examples/01_cosine.py`](examples/01_cosine.py).

Cosine yüksek diye metinler evrensel anlamda benzer değildir. Skor, modelin
eğitim hedefinin ve verisinin tanımladığı uzayda anlamlıdır. Farklı modellerin
skor eşikleri de doğrudan karşılaştırılamaz.

## 2. Chunking stratejileri

Chunk çok küçükse cevap için gerekli iki cümle ayrılabilir; çok büyükse vektör
birden fazla konuyu ortalamaya başlar, alakasız bağlam ve token maliyeti artar.
Tek bir evrensel “doğru chunk boyutu” yoktur.

| Strateji | Ne yapar? | İyi olduğu yer | Temel riski |
| --- | --- | --- | --- |
| Sabit pencere | `N` kelime, `M` overlap | hızlı baseline, düz metin | cümle/başlık ortasından keser |
| Cümle penceresi | cümleleri bütçeye doldurur | SSS, destek, makale | çok uzun tek cümle bütçeyi aşabilir |
| Yapı-duyarlı | Markdown başlığını korur | rehber ve dokümantasyon | kaynak yapısı bozuksa kalite düşer |

[`rag_lab/chunking.py`](rag_lab/chunking.py) bu üçünü uygular;
[`examples/02_chunking.py`](examples/02_chunking.py) aynı metindeki sınırları
yan yana gösterir.

Bu uygulama okunabilirlik için **kelime** sayar. Gerçek sistemde sınırı embedding
modelinin tokenizer'ıyla **token** cinsinden kontrol edin. Başlangıç deneyi için
yaklaşık 200–500 token ve %10–20 overlap denenebilir; bu bir kural değil,
kendi soru kümeniz üzerinde ölçülecek bir hipotezdir.

Chunking deneyinde aynı anda yalnızca bir değişkeni değiştirin ve şunları raporlayın:

- `Recall@k` / MRR ve doğru parçanın sırası;
- indekslenen chunk sayısı ve vektör belleği;
- sorgu gecikmesi ve prompt token sayısı;
- cevabı tek parçada bulunmayan çok-adımlı sorular;
- tablo, kod bloğu, başlık ve dipnot gibi bozulabilen yapılar.

## 3. Embedding modeli seçimi

Bir üretim modelinden beklenen metin yazmasıdır; embedding modelinden beklenen,
arama görevinde ilgili metinleri yakınlaştırmasıdır. Her `AutoModel` çıktısını
ortalayıp iyi bir retrieval modeli elde etmiş olmayız.

Model seçerken en az şunları test edin:

- hedef dil ve alan: Türkçe, kod, hukuk veya çok dillilik;
- **retrieval** ölçümleri; yalnızca genel benchmark ortalaması değil;
- query/document için model kartının istediği prefix veya instruction;
- maksimum giriş uzunluğu ve truncation davranışı;
- embedding boyutu, indeks belleği ve gecikme;
- lisans, çevrimdışı çalışabilme ve donanım bütçesi;
- aynı model sürümünün hem belgede hem sorguda kullanılması.

Bu laboratuvarda üç yol vardır:

| Backend | Amaç | Bağımlılık |
| --- | --- | --- |
| `HashEmbedding` | akışı çevrimdışı öğretme/test etme | yok |
| `OllamaEmbedding` | modeli yerel servis üzerinden çalıştırma | Ollama + çekilmiş model |
| `TransformersEmbedding` | tokenizer, pooling, cihaz ve ağırlıkları koddan yönetme | `torch`, `transformers` |

Transformers örneği attention maskeli mean pooling yapar ve sonucu normalize
eder. Seçilen model farklı pooling ya da özel prompt istiyorsa model kartındaki
tarife uyarlayın.

## 4. Vektör veritabanı ne yapar?

Embedding modeli **vektörü üretir**; vektör deposu onu **saklar ve arar**. En az
şu sözleşme gerekir:

```text
(chunk_id, document_id, vector, raw_text, metadata,
 embedding_model, model_version, dimension, distance_metric)
```

Bu klasörde iki çalışan örnek var:

- `InMemoryVectorStore`: en sade exact cosine taraması; JSON'a model adıyla kaydedebilir.
- `SQLiteVectorStore`: kalıcılık ve metadata'yı gösterir; vektörleri okuyup yine
  Python'da exact tarar, dolayısıyla milyonlarca kayıt için çözüm değildir.

[`examples/03_vector_store.py`](examples/03_vector_store.py) aynı sorguyu iki
depoda çalıştırır ve `category=hesap` metadata filtresini gösterir.

Üretimde seçim kabaca şöyledir:

| İhtiyaç | Başlangıç yaklaşımı |
| --- | --- |
| küçük veri, eğitim, doğruluk referansı | bellekte/NumPy exact search |
| mevcut ilişkisel veri ve SQL filtreleri | vektör uzantılı ilişkisel veritabanı |
| tek makinede gömülü geliştirme | gömülü/local vektör deposu |
| dağıtık, yüksek hacimli retrieval | yönetilen veya bağımsız vektör arama motoru |

HNSW, IVF ve product quantization gibi ANN yöntemleri gecikme/bellek kazanmak
için bazı komşuları kaçırabilir. Bu nedenle ANN sonucunu exact sonuçla karşılaştırıp
**recall–latency–memory** üçlüsünü birlikte ölçün. Model değiştiğinde boyut aynı
kalsa bile uzay değişebilir; eski ve yeni vektörleri aynı koleksiyonda karıştırmayın.

## 5. Ollama ile RAG

Ollama yolu, embedding ve generation modelini yerel HTTP servisi arkasında tutar.
Kod `POST /api/embed` ile batch embedding, `POST /api/generate` ile non-streaming
cevap ister. Resmî API: [embed](https://docs.ollama.com/api/embed) ve
[generate](https://docs.ollama.com/api/generate).

```bash
ollama pull embeddinggemma
ollama pull gemma3

cd rag
python3 -m examples.05_ollama_rag "Standart kargo ne zaman gelir?"
```

Model adlarını değiştirmek için:

```bash
python3 -m examples.05_ollama_rag \
  --embedding-model embeddinggemma \
  --generation-model gemma3 \
  "Parola bağlantısı kaç dakika geçerli?"
```

Ollama'nın güncel dokümanına göre `/api/embed` L2-normalize vektör döndürür;
depo yine de backend sözleşmesini güvenli kılmak için normalize eder.

## 6. Transformers ile RAG

Transformers yolu ağırlıkları aynı Python sürecine yükler. Daha fazla kontrol
sağlar: pooling kodu, batch boyutu, cihaz, dtype ve generation ayarları size
aittir. Buna karşılık model uyumluluğu ve bellek yönetimi de sizin sorumluluğunuzdur.
`AutoModel` / `AutoTokenizer` yükleme davranışı için
[resmî Auto Classes dokümanı](https://huggingface.co/docs/transformers/model_doc/auto)
kullanılmıştır.

```bash
cd rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-optional.txt

python3 -m examples.06_transformers_rag \
  --generation-model MODEL_KARTINDAKI_CAUSAL_LM_ID \
  "Pazarlama iznimi nasıl geri çekerim?"
```

Generation modeli bilinçli olarak zorunlu argümandır: cihazınıza, lisansınıza ve
dilinize uygun instruction-tuned bir causal LM seçmelisiniz. Büyük bir modeli
CPU'ya yanlışlıkla indirmemek için örnek varsayılan koymaz.

### Ollama mı Transformers mı?

| Konu | Ollama | Transformers |
| --- | --- | --- |
| Kurulum | servis + model pull | Python paketleri + model ağırlıkları |
| Kod karmaşıklığı | düşük, HTTP sözleşmesi | daha yüksek, doğrudan model kontrolü |
| Pooling/tokenization | model paketi ve servis yönetir | uygulama açıkça yönetir |
| Donanım/dtype ayarı | kolay varsayılanlar | ayrıntılı kontrol |
| Süreç izolasyonu | model ayrı serviste | uygulamayla aynı süreçte |
| Araştırma/değişiklik | sınırlı ama pratik | katmanlara kadar esnek |

Bu bir kalite sıralaması değildir. Aynı ağırlık, prompt ve sayısal hassasiyet
kullanılmadan backend sonuçlarını “Ollama daha iyi/kötü” diye yorumlamayın.

## 7. Retrieval'dan cevaba

[`rag_lab/pipeline.py`](rag_lab/pipeline.py) şu açık çağrı zincirini kurar:

```python
chunks = chunk_documents(documents)
vectors = embedder.embed([chunk.text for chunk in chunks])
store.add(chunks, vectors)

query_vector = embedder.embed([question])[0]
sources = store.search(query_vector, k=4, filters={"category": "hesap"})
prompt = build_grounded_prompt(question, sources)
answer = generator.generate(prompt)
```

Prompt, bağlam yetersizse bunu söylemeyi, kaynak etiketi kullanmayı ve kaynak
metindeki talimatları uygulamamayı ister. Bu son madde prompt injection riskini
azaltan yalnızca bir katmandır; güvenilmeyen belgeler yetkilendirme yapmamalı,
araç çağrılarını tetiklememeli ve hassas metadata retrieval öncesi filtrelenmelidir.

## 8. Değerlendirme

[`examples/07_evaluate_retriever.py`](examples/07_evaluate_retriever.py) dört
etiketli soruyla `Recall@1`, `Recall@3` ve MRR hesaplar. Hash modeli nedeniyle
skorlar yalnızca mekanizmayı gösterir.

- `Recall@k`: doğru belge ilk `k` adaydan herhangi birinde mi?
- `MRR`: ilk doğru sonucun sırasının tersinin ortalaması; üst sıraları ödüllendirir.
- **Faithfulness/adherence:** üretilen iddialar getirilen bağlamla destekleniyor mu?
- **Answer correctness:** cevap altın cevaba göre doğru mu?
- **Abstention:** bilgi yokken sistem gerçekten “bulunamadı” diyebiliyor mu?

Retriever ve generator değerlendirmesini ayırın. Ardından aynı sorularla uçtan
uca sonucu ölçün. Üretim öncesi en az şunları içeren bir test kümesi hazırlayın:

- kolay tek-parça sorular;
- birden çok parçayı birleştiren sorular;
- cevabı veri tabanında olmayan sorular;
- benzer ama yanlış distractor içeren sorular;
- metadata/erişim filtresi gerektiren sorular;
- Türkçe karakter, yazım hatası ve eşanlamlı varyantlar.

## Dosya rehberi

| Yol | Sorumluluk |
| --- | --- |
| `rag_lab/similarity.py` | dot product, norm, normalize, cosine |
| `rag_lab/chunking.py` | fixed, sentence ve Markdown chunking |
| `rag_lab/embeddings.py` | Hash, Ollama ve Transformers embedding |
| `rag_lab/stores.py` | in-memory/JSON ve SQLite exact depolar |
| `rag_lab/generators.py` | extractive, Ollama ve Transformers generation |
| `rag_lab/pipeline.py` | index, retrieve, grounded prompt, answer |
| `rag_lab/evaluation.py` | Recall@k ve MRR |
| `data/bilgi_bankasi.json` | küçük Türkçe destek bilgi bankası |
| `rag_demo.ipynb` | cosine'dan Ollama/Transformers'a ders notebook'u |
| `examples/` | sıralı, çalıştırılabilir ders demoları |
| `tests/` | bağımlılıksız birim testleri |

## Tartışma soruları

1. Overlap artırıldığında Recall@k yükselirken hangi maliyetler büyür?
2. Cosine ile dot product hangi koşulda tam olarak aynı sıralamayı verir?
3. Doğru chunk retrieval sonucunda üçüncü sıradaysa `k=2` ve `k=10` ne tür
   farklı hatalar doğurur?
4. Embedding modelini güncellerken neden yalnızca yeni sorguları yeni modelle
   embed etmek yetmez?
5. Aynı RAG sistemi için “retrieval doğru, cevap yanlış” ve “retrieval yanlış,
   cevap doğru” durumlarını nasıl ayırırsınız?
6. Kullanıcı A'nın erişemediği bir belge cosine'da birinciyse filtre hangi
   aşamada uygulanmalıdır?
