# Hafta 4 — uygulamalar ve ödev

Alıştırmalar kolaydan zora ilerler. İlk dört görev derste; mini proje haftanın
ödevi olarak yapılabilir.

## Derste

### 1. Cosine'ı elle hesapla

`[1, 1]` ile `[2, 0]` vektörlerinin cosine benzerliğini kâğıt üzerinde bulun,
sonucu [`by_hand.py`](by_hand.py) ile kontrol edin. İki vektörün uzunluğunu iki
katına çıkarmak cosine sonucunu neden değiştirmez?

### 2. Ortalama havuzlamanın kör noktası

Eğitilmemiş `MeanEncoder` ile `ali` ve `ila` embedding'lerini karşılaştırın.
Karakterlerin sırası neden kaybolur? Aynı deneyi `TransformerEncoder` ile yapın
ve RoPE'nin sırayı nasıl görünür hâle getirdiğini açıklayın.

### 3. Negatiflerin rolü

`train.py` içinde batch boyutunu sırasıyla `8`, `32`, `64` yapın. Aynı adım
sayısında loss ve validation `Recall@1` nasıl değişiyor? In-batch negative
sayısının artması neden genellikle faydalıdır, hangi durumda false negative
üretebilir?

### 4. Matryoshka kesmesi

Transformer modelini eğitin. Aynı doğrulama setinde yalnızca ilk `8`, `16` ve
`32` boyutu kullanarak retrieval `Recall@1` hesaplayın. Her kesitten sonra
vektörü yeniden normalize etmeyi unutmayın.

## Haftanın ödevi — özel doküman indeksleme

Kendinize ait, paylaşma hakkınız bulunan en az 100 kısa metin seçin. Her kayıt
şu yapıda olsun:

```json
{"id": "doc-001", "text": "...", "source": "..."}
```

Teslimde şunlar bulunmalı:

1. `dataset.jsonl` — temizlenmiş dokümanlar ve kaynak metadatası.
2. `embed.py` — batch hâlinde embedding üretimi.
3. `index.py` — vektör ve metadata ekleme; en az `top_k` araması.
4. `evaluate.py` — en az 20 sorgu, beklenen doküman id'si ve `Recall@1/5`.
5. `README.md` — model, boyut, chunk yöntemi, distance metric ve gözlenen iki
   hata örneği.

### Asgari kabul ölçütleri

- Train/test ayrımı veri hazırlanmadan önce sabit seed ile yapılmış olmalı.
- Aramada kullanılan doküman ve sorgu vektörleri aynı normalizasyon kuralına
  sahip olmalı.
- Sonuç yalnızca ekran görüntüsü değil, tekrar çalıştırılabilir kod olmalı.
- Kişisel veya lisanssız veri repoya konmamalı.
- En az bir lexical baseline (basit karakter örtüşmesi dahi olabilir) ile
  embedding sonucu karşılaştırılmalı.

### Bonuslar

- Exact search ile HNSW tabanlı ANN aramayı hız/kalite açısından kıyaslayın.
- Tek-vektörlü arama ile late interaction sonuçlarını karşılaştırın.
- Metadata filtresi ekleyin.
- Türkçe karakterleri ASCII'ye çevirmenin hangi sorguları düzelttiğini, hangilerini
  bozduğunu örnekleyin.
