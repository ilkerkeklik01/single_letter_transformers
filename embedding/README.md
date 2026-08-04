# Hafta 4 — Vektör Uzayı ve Temsil Öğrenimi

Bu modül, “embedding nedir?” sorusunu tek bir API çağrısıyla geçmek yerine üç
küçük mimariyi **sıfırdan kurar, eğitir, ölçer ve bir arama indeksine bağlar**.
Hepsi aynı oyuncak problemi kullanır: Türkçe harfleri ve adları yazım
benzerliğine göre bir vektör uzayına yerleştirmek.

> Amaç büyük bir genel amaçlı embedding modelini taklit etmek değil; modern
> sistemlerdeki kararları gözle görülebilir hâle getirmektir. Buradaki “benzerlik”
> eğitim verisi nedeniyle ağırlıklı olarak **yazımsal** benzerliktir. “Ali” ile
> “Veli”nin dünyadaki anlam ilişkisini, yalnızca isim listesinden öğrenemeyiz.

> Mimarileri tensor şekilleri, denklemler, EOS/mean/MaxSim pooling farkları,
> multimodal patch akışı ve üretim tuzaklarıyla birlikte okumak için:
> **[Embedding Mimarileri — Derinlemesine Ders Rehberi](MIMARI_REHBERI.md)**

## Haftanın çıktıları

Bu haftanın sonunda öğrenci:

- token embedding ile metin embedding arasındaki farkı açıklar;
- dot product, cosine ve L2 ölçülerini doğru yerde kullanır;
- dense bi-encoder ile late-interaction yaklaşımını karşılaştırır;
- positive pair, in-batch negative ve sıcaklık değerinin görevini açıklar;
- Matryoshka hedefiyle tek modeli birden çok embedding boyutunda eğitir;
- embedding'leri indeksler, `top-k` arar ve `Recall@k` ile ölçer;
- vektör veritabanının modelden ayrı bir sistem bileşeni olduğunu bilir.

## Ders planı

### Gün 1 — Embedding modelleri: nedir, nasıl çalışır ve nasıl eğitilir? (2 saat)

| Süre | Konu | Somut çıktı |
| ---: | --- | --- |
| 0–20 dk | Vektör uzayı sezgisi; yön, uzunluk, cosine | `by_hand.py` §1 |
| 20–40 dk | Token, metin ve çoklu-vektör temsilleri | şekil takibi |
| 40–65 dk | Mean pooling baseline ve sıra kaybı | `mean_encoder.py` |
| 65–95 dk | Çift yönlü Transformer encoder | `transformer_encoder.py` |
| 95–110 dk | Contrastive InfoNCE ve in-batch negatives | `losses.py` |
| 110–120 dk | Matryoshka boyutları: 8 → 16 → 32 | kısa deney |

### Gün 2 — Vektör veritabanı ve projeye entegrasyon (2 saat)

| Süre | Konu | Somut çıktı |
| ---: | --- | --- |
| 0–25 dk | Embed → normalize → index → query akışı | sistem sözleşmesi |
| 25–50 dk | Exact cosine search ve `top-k` | `vector_store.py` |
| 50–75 dk | ANN sezgisi; HNSW/IVF/PQ ve kalite-hız dengesi | tasarım tartışması |
| 75–95 dk | ColBERT-benzeri MaxSim ile late interaction | `late_interaction.py` |
| 95–110 dk | `Recall@1/5`, false negative ve veri sızıntısı | değerlendirme |
| 110–120 dk | Ödev veri sözleşmesi ve teslim kriterleri | `EXERCISES.md` |

## Bir embedding gerçekte nedir?

Bir encoder, değişken uzunluktaki metni sabit boyutlu bir vektöre dönüştürür:

```text
"çağla" -> encoder -> [0.12, -0.03, ..., 0.41]  (32 sayı)
```

İki normalize vektör için dot product ile cosine aynıdır:

```text
cos(a, b) = (a · b) / (||a|| ||b||)
||a|| = ||b|| = 1 ise cos(a, b) = a · b
```

Cosine `+1` yönlerin aynı, `0` dik, `-1` zıt olduğunu söyler. Bu tek başına
“anlam” değildir; model hangi çiftleri yakınlaştırarak eğitildiyse uzay onu
kodlar.

## Üç mimari, aynı görev

| Mimari | Çıktı | Güçlü yanı | Bilerek görünen sınırı |
| --- | --- | --- | --- |
| `mean` | ad başına 1×32 | çok hızlı, 8.224 parametre | harf sırasını kaybeder |
| `transformer` | ad başına 1×32 | sıra ve bağlamı modeller | bütün bilgi tek vektöre sıkışır |
| `late` | harf başına 1×32 | ince token-token eşleşmesi | indeks ve arama daha pahalıdır |

Model-ailelerine özgü devam uygulamaları ayrı ve kendi başına okunabilir
klasörlerdedir:

- [`../qwen3_embedding/`](../qwen3_embedding/) — Qwen3-Embedding'in causal
  attention, instruction token, son EOS pooling ve MRL yolu.
- [`../embedding_gemma/`](../embedding_gemma/) — EmbeddingGemma'nın bidirectional
  encoder, mean pooling, iki lineer projection ve GOR yolu.
- [`../qwen3_vl_embedding/`](../qwen3_vl_embedding/) — Qwen3-VL-Embedding'in
  vision patch/merger, ortak text-image uzayı ve multimodal dual tower yolu.

### 1. MeanEncoder — dürüst baseline

```text
token embedding -> PAD'leri maskele -> ortala -> MLP -> L2 normalize
```

Bu model `ali` ve `ila` için aynı karakterlerin ortalamasını görür. Baseline'ın
kusuru dersin parçasıdır: daha karmaşık modelin neyi çözdüğünü ancak basit bir
referansla anlayabiliriz.

### 2. TransformerEncoder — modern çift-kule yolu

Aynı encoder hem anchor hem positive için tekrar kullanılır; yani iki kulenin
ağırlıkları ortaktır:

```text
anchor   -> shared encoder -> u ┐
                                ├─ cosine matrix -> InfoNCE
positive -> shared encoder -> v ┘
```

Encoder **causal değildir**; her harf sağını ve solunu görebilir. Küçük blokta
güncel Transformer tariflerinde sık görülen RMSNorm, QK-Norm, RoPE, pre-norm ve
SwiGLU kullanılır. Son token'lar öğrenilen attention pooling ile tek vektöre
iner. Bu bir BGE/E5/Qwen kopyası değil; ortak tasarım fikirlerinin okunabilir
bir öğretim bileşimidir.

### 3. LateInteractionEncoder — çoklu vektör ve MaxSim

Dense encoder bütün adı tek vektöre sıkıştırır. Late interaction ise her harfin
bağlamsal vektörünü saklar. Bir sorgu harfi, dokümandaki en iyi eşini seçer;
sonra sorgu boyunca ortalama alınır:

```text
score(q, d) = mean_i max_j cosine(q_i, d_j)
```

Örnek şekiller:

```text
queries   [B, Tq, 32]
documents [C, Td, 32]
all pairs [B, C, Tq, Td]
MaxSim    [B, C]
```

Bu, ColBERT'in ana fikrinin küçük ve bağımlılıksız bir uygulamasıdır. Gerçek
ColBERT indeksleme, sıkıştırma ve özel retrieval altyapısına sahiptir; burada
yalnızca mimari ayrımı görünür kılınır.

## Model nasıl öğreniyor?

Etiketli benzer isim çiftimiz olmadığı için iki tür positive üretiyoruz:

1. Bir ismin tek edit uygulanmış görünümü: silme, komşu harf değişimi,
   transpozisyon veya Türkçe karakteri ASCII'ye katlama.
2. Açıkça tanımlanmış küçük harf çiftleri: `c–ç`, `g–ğ`, `o–ö`, `u–ü` gibi.

Batch içindeki diğer örnekler negative olur. `B` çift için model `B × B` skor
matrisi üretir; doğru eşler köşegendedir. InfoNCE köşegeni büyütüp diğer hücreleri
küçültür:

```text
logits[i,j] = cosine(anchor_i, positive_j) / temperature
label[i]    = i
```

Kayıp iki yönde hesaplanır. `temperature=0.07`, küçük skor farklarını daha
keskin hâle getirir. Batch'te gerçekten benzer iki farklı isim varsa bunlar
yanlışlıkla negative sayılabilir; buna **false negative** denir.

### Matryoshka hedefi

Dense modellerde aynı contrastive kayıp ilk `8`, ilk `16` ve tüm `32` boyutta
ayrı ayrı hesaplanıp ortalanır:

```text
L = mean(L(z[:8]), L(z[:16]), L(z[:32]))
```

Böylece inference sırasında 32 boyutlu vektörü yeniden eğitim yapmadan 16 veya
8 boyuta kesebiliriz. Kesilen vektörü cosine öncesinde tekrar normalize etmek
gerekir. Bu, Matryoshka Representation Learning fikrinin küçük metin karşılığıdır.

## Çalıştırma

Komutları bu klasörde çalıştırın:

```bash
cd embedding

# Sayıları elle izleyin; eğitim yok.
python3 by_hand.py

# Üç modeli ayrı ayrı eğitin.
python3 train.py mean --steps 400
python3 train.py transformer --steps 800
python3 train.py late --steps 800

# Harf/isim çiftlerini puanlayın ve en yakın isimleri bulun.
python3 demo.py tiny_transformer_embedding.pt ayse mehmed çağla

# Tek-vektörlü bir checkpoint ile kalıcı exact-cosine indeks kurun.
python3 build_index.py tiny_transformer_embedding.pt

# Testler.
python3 -m unittest discover -s tests -v
```

CPU'da yapılan kısa bir geliştirme koşusunda (`seed=1337`, 120 adım, batch 32)
`mean` için validation `Recall@1 ≈ 0.94`, `transformer` için `≈ 0.92` görüldü.
Bu değerler yayımlanmış benchmark değildir; donanım/sürüm ve adım sayısına göre
değişebilen smoke-test sonuçlarıdır. Asıl hedef aynı split üzerinde modellerin
göreli davranışını incelemektir.

## Vektör veritabanı: model değildir

[`vector_store.py`](vector_store.py) tüm normalize doküman vektörlerini bir
matriste tutar ve sorguyu bu matrisle çarpar. Bu **exact search**'tür:

```text
metinler -> embedding modeli -> normalize vektörler -> INDEX
sorgu    -> aynı model        -> normalize vektör  -> top-k
```

921 isim için bu yöntem ideal derecede basit ve yeterince hızlıdır. Milyonlarca
dokümanda bütün satırları taramak pahalılaşır; üretim sistemleri HNSW veya IVF
gibi approximate nearest-neighbour (ANN) indeksleri ve bazen product
quantization kullanır. ANN hız ve bellek kazanırken az miktarda recall kaybedebilir.

Vektör veritabanının saklaması gereken sözleşme en az şunlardır:

- vektör ve benzersiz doküman id'si;
- model adı/sürümü ve embedding boyutu;
- kullanılan normalizasyon ve distance metric;
- kaynak metadata ve erişim filtresi;
- yeniden indeksleme için ham metne güvenli referans.

Model değişirse eski ve yeni vektörleri sessizce karıştırmayın. Boyut aynı olsa
bile uzay aynı değildir; sürümlü yeni bir koleksiyon oluşturun.

## Değerlendirme ve beklenen tuzaklar

- **Train/test sızıntısı:** Aynı adın bozuk görünümü iki split'e dağılmamalı.
- **Yanlış metrik:** Cosine kullanılacaksa iki taraf da aynı şekilde normalize edilmeli.
- **Sadece güzel örnekler:** Üç başarılı nearest-neighbour çıktısı benchmark değildir.
- **Kimlik ezberleme:** Validation, eğitimde görülmeyen temiz adlardan kurulmalıdır.
- **Unicode:** `ş` ile `s + combining cedilla` farklı kod noktalarıdır. Tokenizer
  ikisini NFC'ye çevirir; bu adım kaldırılırsa görsel olarak aynı sorgular ayrışır.
- **Anlam iddiası:** Bu veri adların anlamını, kökenini veya kişileri öğretmez.
- **Late-interaction maliyeti:** Bir doküman yerine her token için vektör saklanır.

Bu modüldeki `retrieval_at_one`, bozulmuş bir validation adının kendi temiz
sürümünü aday havuzunda birinci getirip getirmediğini ölçer. Gerçek ödevde ayrıca
`Recall@5`, latency, indeks boyutu ve nitel hata analizi raporlanmalıdır.

## Dosya rehberi

| Dosya | Sorumluluk |
| --- | --- |
| `config.py` | tüm model boyutları ve Matryoshka kesitleri |
| `tokenizer.py` | harf tokenizasyonu, PAD/UNK ve batch maskesi |
| `augment.py` | positive isim/harf çiftleri |
| `attention.py`, `block.py`, `mlp.py` | çift yönlü modern encoder bloğu |
| `mean_encoder.py` | sıra-bilmez tek-vektör baseline |
| `transformer_encoder.py` | bağlamsal tek-vektör bi-encoder |
| `late_interaction.py` | token vektörleri ve MaxSim |
| `losses.py` | symmetric InfoNCE ve Matryoshka kaybı |
| `train.py` | ortak eğitim ve validation akışı |
| `vector_store.py`, `build_index.py` | exact cosine indeksleme |
| `demo.py` | harf/isim benzerliği ve nearest-neighbour arama |
| `by_hand.py` | kâğıtla takip edilebilir sayılar |
| `EXERCISES.md` | sınıf uygulamaları ve haftanın ödevi |
| `MIMARI_REHBERI.md` | bütün modellerin ayrıntılı, karşılaştırmalı ders anlatımı |

## Bilerek sadeleştirilenler

- Veri yalnızca 921 addan oluşur; anlamsal cümle embedding'i beklenmez.
- Hard-negative mining, çapraz cihaz batch'i ve distillation yoktur.
- Tokenizer karakter seviyesindedir; gerçek modeller subword tokenizasyon kullanır.
- Late-interaction indeksi sıkıştırılmaz ve özel kernel kullanmaz.
- `VectorIndex`, transaction/replication/filtering sunan üretim veritabanı değildir.
- Harf çiftleri öğretim amaçlı sezgisel supervision'dır; fonetik veri seti değildir.

## Birincil kaynaklar

- Reimers & Gurevych, [Sentence-BERT](https://arxiv.org/abs/1908.10084) — ortak
  encoder ve cosine ile verimli cümle karşılaştırma.
- Gao, Yao & Chen, [SimCSE](https://arxiv.org/abs/2104.08821) — contrastive
  cümle embedding eğitimi.
- Khattab & Zaharia, [ColBERT](https://arxiv.org/abs/2004.12832) — bağlamsal
  late interaction ve MaxSim.
- Kusupati et al., [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
  — kesilebilir, coarse-to-fine temsiller.
- Chen et al., [BGE M3-Embedding](https://arxiv.org/abs/2402.03216) — dense,
  sparse ve multi-vector retrieval yollarını bir arada ele alan güncel örnek.
