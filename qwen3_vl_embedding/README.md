# Tiny Qwen3-VL-Embedding

Qwen3-VL-Embedding'in **multimodal dual-tower + EOS pooling** fikrini Türkçe ad
görüntüleriyle gösteren küçük uygulama.

Patch tensor şekilleri, 2×2 spatial merger, ortak text/image uzayı ve fused input
akışı için ana ders notundaki
[Qwen3-VL-Embedding bölümünü](../embedding/MIMARI_REHBERI.md#9-qwen3-vl-embedding-text-ve-image-ortak-uzayda)
okuyun.

```text
text  -> character tokens ---------------------┐
                                                ├-> shared causal Qwen -> EOS -> vector
image -> patches -> vision encoder -> 2x2 merge┘
```

Text ve image ayrı ayrı encode edilebildiği için belge görüntüleri önceden
indekslenebilir. `encode_fused()` ise image+text karışımını tek vektöre çevirir.
Her yol aynı projection ve `8/16/32` Matryoshka boyutlarını kullanır.

## Mimaride görünür tuttuğumuz güncel fikirler

- Vision patch encoder ve **2×2 spatial patch merger**.
- Görsel/text token'larının ortak causal Qwen omurgasına girmesi.
- Her girdinin sonuna öğrenilen `<EOS>` ekleyip son hidden state'i alma.
- Text–image symmetric contrastive loss ile ortak uzay.
- Dual tower ile hızlı ilk retrieval; reranker bu eğitim modülünde yoktur.

Resmî seri Qwen3-VL tabanlı 2B/8B modeller, 32K bağlam, text/image/screenshot/video
ve mixed-modal input, instruction awareness ve esnek embedding boyutları sunar.
Buradaki model tek kanallı 32×96 yazı görüntüleri kullanır; video/screenshot
kalitesi, resmî tokenizer veya checkpoint uyumluluğu iddiası yoktur.

## Çalıştırma

Pillow yalnızca oyuncak isim görsellerini üretmek için gerekir:

```bash
cd qwen3_vl_embedding
python3 train.py --steps 600
python3 demo.py
python3 -m unittest discover -s tests -v
```

`render.py`, her temiz ismi küçük grayscale görüntüye çevirir. Eğitim ASCII'ye
katlanmış text sorgusunu doğru isim görüntüsüyle eşler; batch'teki diğer
görüntüler negative olur.

## Kaynaklar

- Qwen Team, [Qwen3-VL-Embedding teknik raporu](https://arxiv.org/abs/2601.04720)
- [Resmî Qwen3-VL-Embedding deposu](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [Resmî 8B model kartı](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)
