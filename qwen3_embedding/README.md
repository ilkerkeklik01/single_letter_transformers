# Tiny Qwen3-Embedding

Qwen3-Embedding'in ayırt edici metin-embedding yolunu küçük Türkçe isim görevi
üzerinde gösteren, yalnızca PyTorch kullanan öğretim implementasyonu.

EOS pooling'in neden çalıştığı, causal görünürlük matrisi, instruction token'ları
ve Matryoshka eğitim akışı için ana ders notundaki
[Qwen3-Embedding bölümünü](../embedding/MIMARI_REHBERI.md#6-qwen3-embedding-causal-attention--eos-pooling)
okuyun.

```text
<query/document> + karakterler + <eos>
        -> causal Qwen3 decoder
        -> son <eos> hidden state
        -> projection -> MRL kesiti -> L2 normalize
```

## Gerçek modelden alınan fikirler

- **Causal decoder + EOS pooling:** Çift yönlü encoder yerine Qwen3 LLM omurgası
  ve son token temsili kullanılır.
- **Instruction aware:** Sorgu ve doküman farklı görev token'ıyla başlar.
- **Qwen bloğu:** GQA, QK-Norm, RoPE, RMSNorm ve SwiGLU.
- **MRL:** Aynı vektör 8/16/32 boyutta eğitilir ve inference'ta kesilebilir.
- **Dual tower:** Query ve document bağımsız, aynı ağırlıklarla encode edilir.

Resmî Qwen3-Embedding ailesi 0.6B/4B/8B boyutlarında, 32K bağlamlı ve instruction
aware modeller sunar. Bu klasör 32 boyutlu, yaklaşık 80 bin parametreli bir
mekanik modeldir; resmî ağırlıkların küçültülmüş checkpoint'i veya kalite eşdeğeri
değildir.

## Çalıştırma

```bash
cd qwen3_embedding
python3 train.py --steps 800
python3 demo.py
python3 -m unittest discover -s tests -v
```

`demo.py` aynı sorguları 8, 16 ve 32 boyutta aratır. Sorgu ve dokümanı yanlış rol
token'ıyla encode etmek, instruction-aware modellerde sessiz kalite kaybına yol
açan gerçek bir entegrasyon hatasını görünür kılar.

## Dosyalar

`config → rms_norm → rotary → attention → mlp → block → model`; `tokenizer.py`
rol/EOS sözleşmesini, `losses.py` Matryoshka InfoNCE'yi, `train.py` ise aynı isim
listesindeki noisy-query retrieval eğitimini taşır.

## Bilerek sadeleştirilenler

- Karakter tokenizer, iki katman ve 32 token bağlam kullanılır.
- Büyük ölçekli unsupervised pretraining, hard-negative mining, model merging ve
  reranker yoktur.
- Instruction doğal dil prompt'u yerine öğrenilen tek bir rol token'ıdır.

## Kaynak

- Qwen Team, [Qwen3 Embedding teknik raporu](https://arxiv.org/abs/2506.05176)
- [Resmî Qwen3-Embedding-8B model kartı](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
