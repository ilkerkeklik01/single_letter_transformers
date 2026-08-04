# Tiny EmbeddingGemma

Google'ın 2025'te yayımladığı EmbeddingGemma tarifinin okunabilir, küçük bir
mekanik karşılığıdır.

Bidirectional/causal farkı, mean pooling, aktivasyonsuz projection neck ve GOR
geometrisi için ana ders notundaki
[EmbeddingGemma bölümünü](../embedding/MIMARI_REHBERI.md#7-embeddinggemma-bidirectional--mean-pooling)
okuyun.

```text
bidirectional Gemma encoder -> masked mean pool
-> Linear(64, 256) -> Linear(256, 32) -> L2 normalize
```

## Neyi özellikle gösteriyor?

- Decoder-only Gemma'dan farklı olarak **çift yönlü encoder attention**.
- Token çıktılarında **mean pooling**.
- Arada aktivasyon olmayan `hidden → 4×hidden → embedding` lineer boyun.
- `NCE + Global Orthogonal Regularizer` birleşimi.
- `8/16/32` boyutlu Matryoshka hedefi.
- İç blokta Gemma ailesini hatırlatan sandwich RMSNorm, GeGLU, GQA ve
  local/global attention.

Resmî EmbeddingGemma 308M parametre, 2K context, 768 boyut (128/256/512'ye
kesilebilir) ve 100+ dil desteğiyle on-device kullanım için tasarlanmıştır. Bu
klasör aynı kalitede veya checkpoint uyumlu değildir; yaklaşık 100 bin parametreli
bir ders modelidir.

```bash
cd embedding_gemma
python3 train.py --steps 800
python3 demo.py
python3 -m unittest discover -s tests -v
```

`losses.py` içindeki GOR, normalize embedding'lerin batch ikinci momentini
`I / dimension` hedefine ve ortalamayı sıfıra iter. Bu, küresel uzaya yayılma
fikrinin küçük bir uygulamasıdır; resmî eğitim tarifindeki geometric distillation,
model souping ve QAT burada uygulanmaz.

## Kaynaklar

- Google DeepMind, [EmbeddingGemma teknik raporu](https://arxiv.org/abs/2509.20354)
- Google Developers, [EmbeddingGemma architecture and recipe](https://developers.googleblog.com/en/gemma-explained-embeddinggemma-architecture-and-recipe/)
- [Resmî model kartı](https://ai.google.dev/gemma/docs/embeddinggemma/model_card)
