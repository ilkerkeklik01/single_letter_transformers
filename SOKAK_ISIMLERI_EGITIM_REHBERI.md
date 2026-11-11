# Sokak isimleriyle model eğitimi

Bu rehber, 50 Türkçe sokak/cadde adıyla küçük karakter-seviyeli Qwen3 modelini
eğitmek için gereken adımları anlatır. Model kodlarına dokunulmaz; yalnızca veri
hazırlama ve eğitim scripti kullanılır.

## 1. Projeye girin

```bash
cd /Users/ilkerkeklik/Desktop/magibu_ai_course_contents/single_letter_transformers
```

## 2. Python 3.13.14 sanal ortamını oluşturun

`.venv` daha önce oluşturulmadıysa:

```bash
~/.pyenv/versions/3.13.14/bin/python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch numpy
```

Sanal ortamı etkinleştirin:

```bash
source .venv/bin/activate
python --version
```

Beklenen sürüm:

```text
Python 3.13.14
```

## 3. Ham veriyi hazırlayın

Ham sokak isimleri [`data/sokak_isimleri.txt`](data/sokak_isimleri.txt)
dosyasındadır. Her satırda bir sokak veya cadde adı bulunur. Yeni kayıt eklemek
isterseniz bu dosyaya ekleyebilirsiniz.

## 4. Veriyi temizleyin

[`data/temizle_sokak_isimleri.py`](data/temizle_sokak_isimleri.py) scripti
Türkçe büyük/küçük harf dönüşümü yapar, fazla boşlukları temizler ve tekrarları
siler:

```bash
python data/temizle_sokak_isimleri.py
```

Çıktı [`data/temiz_sokak_isimleri.txt`](data/temiz_sokak_isimleri.txt)
dosyasına yazılır. Sokak adlarının içindeki boşluklar korunur.

Kayıt sayısını kontrol edin:

```bash
wc -l data/sokak_isimleri.txt
wc -l data/temiz_sokak_isimleri.txt
```

## 5. Modeli eğitin

[`qwen3/train.py`](qwen3/train.py) temiz sokak verisini okuyacak şekilde
ayarlanmıştır:

```bash
cd qwen3
python train.py
```

Eğitim sonunda checkpoint şu dosyaya kaydedilir:

```text
qwen3/tiny_qwen.pt
```

Bu dosya her eğitimde yeniden yazılır.

## 6. Yeni sokak isimleri üretin

Eğitim tamamlandıktan sonra `qwen3` klasöründe:

```bash
python generate.py 20 0.8
```

Burada `20` üretilecek sonuç sayısı, `0.8` ise temperature değeridir.
Daha kontrollü sonuçlar için `0.6`, daha çeşitli sonuçlar için `1.0`
kullanılabilir:

```bash
python generate.py 20 0.6
python generate.py 20 1.0
```

## Baştan sona kısa komutlar

```bash
cd /Users/ilkerkeklik/Desktop/magibu_ai_course_contents/single_letter_transformers
source .venv/bin/activate
python data/temizle_sokak_isimleri.py
cd qwen3
python train.py
python generate.py 20 0.8
```

50 kayıt küçük bir veri setidir. Bu nedenle model bazen mevcut isimleri
tekrarlayabilir veya yarım/anlamsız isimler üretebilir. Daha iyi sonuçlar için
daha fazla sokak adı eklenmelidir.

Sanal ortamdan çıkmak için:

```bash
deactivate
```
