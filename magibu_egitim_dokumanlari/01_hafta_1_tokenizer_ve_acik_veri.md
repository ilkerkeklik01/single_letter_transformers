# Hafta 1: Tokenizer Mimarisi, LLM Çalışma Temelleri ve Açık Veri Ekosistemi

Bu doküman, Magibu Akademi Uygulamalı Yapay Zekâ Mimarisi Eğitimi'nin 1. haftasında işlenen tüm teknik konuları, teorik temelleri, canlı ders detaylarını ve ödev çözümlerini içermektedir.

---

## 📺 İlgili Video Kaynakları
- 🎥 [Hafta 1, Gün 1: Tokenizer Mimarisi ve LLM Temelleri](https://www.youtube.com/watch?v=IzL_TzjXR1s)
- 🎥 [Hafta 1: Ödev Çözümleri, BPE Tokenizer ve LoRA Fine-Tuning](https://www.youtube.com/watch?v=SpNk8XWwEe8)

---

## 1. LLM Çalışma Temelleri ve Tokenizasyon

Büyük Dil Modelleri (LLM'ler) ham metinleri doğrudan sayısal matris işlemlerine alamazlar. Metinlerin matematiksel temsiline geçilmeden önceki ilk adım **tokenizasyon** işlemidir.

### Temel Kavramlar:
1. **Token:** Bir kelime, kelime kökü, ek, hece, karakter veya özel bir kontrol simgesi (`<eos>`, `<bos>`, `<think>`, `<|im_start|>`) olabilir.
2. **Vocab Size (Sözlük Boyutu):** Modelin tanıdığı toplam benzersiz token sayısıdır.
   - Gemma: 256,000 token
   - Qwen 3 / Qwen 3.5: 151,643 ~ 248,000 token
   - Llama 3: 128,000 token
   - Minyatür Modeller (Eğitim Amaçlı): 30 - 512 token
3. **Token ID:** Her token'ın sözlükteki benzersiz integer indeksidir. (Örn: GEMA sözlüğünde `2733` -> `" program"`).

### Tokenizasyon Seviyeleri ve Karşılaştırma:
- **Character-Level (Karakter Seviyesi):** Sözlük boyutu küçüktür (29 Türkçe harf + semboller = ~30). Ancak dizi uzunluğu çok uzar, dizilimlerin anlamsal ilişkisini öğrenmek işlemci maliyetini artırır.
- **Word-Level (Kelime Seviyesi):** Sözlük devasadır (yüz binlerce kelime). Türkçe gibi eklemeli (agglutinative) dillerde yeni türetilen kelimeler sözlükte bulunamaz (*Out-of-Vocabulary / OOV* problemi).
- **Subword-Level (Alt-Kelime Seviyesi):** Günümüz LLM'lerinin standart yöntemidir. BPE (Byte-Pair Encoding), WordPiece ve Unigram algoritmaları ile kök ve ekleri frekans analizine göre optimize ederek parçalar.

---

## 2. Byte-Pair Encoding (BPE) Algoritması Çalışma Mantığı

BPE, verideki frekansa dayalı olarak ardışık karakter veya bayt çiftlerini birleştirerek sözlük oluşturan yinelemeli bir algoritmadır:

1. Metindeki tüm tekil karakterler/baytlar başlangıç sözlüğüne eklenir.
2. Metin boyunca yan yana en çok tekrar eden çift (*byte pair*) tespit edilir.
3. Bu çift birleştirilerek yeni bir token olarak sözlüğe eklenir (*merge rule*).
4. Belirlenen `vocab_size` boyutuna ulaşılana kadar adımlar tekrarlanır.

---

## 3. Türkçe ve Eklemeli Dillerde Morfolojik Tokenizasyon

İngilizce ağırlıklı BPE tokenizer'lar Türkçe kelimeleri anlamsız parçalara bölebilir (Örn: `üreterek` -> `ü + rete + rek`). 
Morfolojik olarak farkında (*linguistically-informed / morphologically-aware*) hazırlanan tokenizer'lar ise kök ve ek ayrımını doğru yaparak (`üret + erek`) dil modelinin anlamsal ilişkileri ve gramer yapısını katbekat daha hızlı öğrenmesini sağlar.

---

## 4. Hafta 1 Ödevi ve Kod Örnekleri

Hafta 1 ödevi 3 temel aşamadan oluşur:

### 1. Aşama: Alan Odaklı Veri Kümesi (Domain-Specific Dataset) Hazırlama
Hugging Face `datasets` kütüphanesi kullanarak Türkçe/İngilizce konuşma/diyalog formatında dataset oluşturup Hub'a yükleme.

```python
import json
from datasets import Dataset, DatasetDict

# Diyalog ve Thinking formatına uygun veri yapısı
data_tr = {
    "train": [
        {
            "conversations": [
                {"role": "user", "content": "Piyasadaki %100 meyve sularında katkı maddesi var mı?"},
                {"role": "thinking", "content": "Kullanıcı gıda güvenliği ve etiket bilgisi sormaktadır."},
                {"role": "assistant", "content": "%100 meyve suyu etiketli ürünlerde ilave şeker veya koruyucu bulunmaz."}
            ]
        }
    ]
}

# Hugging Face Dataset formatına çevirme ve yükleme
dataset = Dataset.from_list(data_tr["train"])
dataset_dict = DatasetDict({"train": dataset})
dataset_dict.push_to_hub("kullanici_adi/my-domain-dataset", private=False)
```

### 2. Aşama: BPE Tokenizer Eğitimi
Hugging Face `tokenizers` kütüphanesi ile metin külliyatı üzerinden BPE eğitip Hub'a yükleme:

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(vocab_size=8192, special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)

tokenizer.save("tokenizer.json")
tokenizer.model.save(".", "my_bpe_tokenizer")
```

### 3. Aşama: Unsloth & LoRA ile Model İnce Ayarı (Fine-Tuning)
Google Colab (T4 GPU) üzerinde `Unsloth` ile Gemma 3 / Qwen modeli üzerine LoRA fine-tuning yapma ve adaptörleri yayımlama.
