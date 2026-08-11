# Hafta 2: Ölçülebilirlik (Benchmarks) ve Model Özelleştirme (LoRA/QLoRA)

Bu doküman, Magibu Akademi Uygulamalı Yapay Zekâ Mimarisi Eğitimi'nin 2. haftasında işlenen tüm teknik konuları, teorik temelleri ve benchmark ölçüm metotlarını içermektedir.

---

## 📺 İlgili Video Kaynakları
- 🎥 [Hafta 2, Gün 1: Ölçülebilirlik ve Model Özelleştirme](https://www.youtube.com/watch?v=R-gww-r6zd0)

---

## 1. Benchmark (Ölçek / Kıyaslama) Nedir ve Neden Kritiktir?

Mühendisliğin bel kemiği ölçümdür (*"Ölçemediğin bir şeyi geliştiremezsin"*). Yapay zekâ modellerinin başarısını, bilgisini, akıl yürütme becerisini ve talimat takip etme yeteneğini objektif olarak ölçen test sistemlerine **Benchmark** denir.

### Önemli Benchmark Çeşitleri ve Ölçekler:
1. **MMLU (Massive Multitask Language Understanding):** 57 faklı akademiki/profesyonel alanda (tıp, hukuk, felsefe, matematik vb.) çoktan seçmeli sorularla modelin bilgisini ölçer.
2. **MMLU-Pro:** MMLU'nun 10 şıklı versiyonudur. Rastgele sallama ihtimalini düşürür.
3. **Magibu Bench (Türkçe MMLU):** ÖSYM, TUS, ALES, Anadolu Üniversitesi çıkmış sınav sorularından derlenmiş 62 farklı kategoride 6,200+ soruluk şeffaf Türkçe bilgi benchmark'ı.
4. **GSM8K & MATH:** Çok adımlı matematiksel problem çözme yeteneği.
5. **HumanEval & SWE-bench:** Kod yazma ve yazılım mühendisliği (Issue çözme) becerileri.
6. **Terminal Bench:** Modelin terminalde komut çalıştırma ve sistem yönetme becerisi.
7. **LMSYS Chatbot Arena:** Çift kör (double-blind) insan oylaması ile modellerin Elo puanlarını belirleyen en güvenilir canlı liderlik tablosu.

---

## 2. Benchmark Çalıştırma ve Regex / Format Yakalama Zorlukları

Modelleri benchmark testine sokarken yanıt biçimleri sorun oluşturabilir:
- Model doğrudan `E` demek yerine `E) Max Weber` veya `Doğru cevap E seçeneğidir` yazabilir.
- **Regex ve Semantik Eşleme:** Model çıktısından seçeneği ayıklamak için Regex ifadeleri veya anlamsal benzerlik (*semantic similarity*) kullanılır.
- **Reasoning / Thinking Modellerinde Max Tokens:** Reasoning modellerinde `max_tokens=1` yapılırsa model düşünme adımını tamamlayamadan kesilir (`null` döner).

---

## 3. Model Özelleştirme: LoRA, QLoRA ve PEFT

- **Full Fine-Tuning:** Tüm parametreleri eğitir. Devasa GPU belleği gerektirir.
- **LoRA (Low-Rank Adaptation):** Dondurulmuş katmanların yanına $A 	imes B$ düşük rütbeli matrisler ekleyerek eğitilecek parametre sayısını %99 oranında azaltır (Örn: 704 parametre).
- **QLoRA:** Ana modeli 4-bit NormalFloat (NF4) quantization ile dondurup LoRA adaptörünü eğitir.

---

## 4. Hafta 2 Ödevi ve Uygulama Adımları

1. **Benchmark Çalıştırma:**
   - Hazır benchmark kodları ile kendi eğittiğiniz modeli ve bir baseline modeli (Örn: Gemma / Qwen) MMLU veya özel test kümesinde değerlendirme.
2. **Model Doğrulama (Identity Fine-Tuning Verification):**
   - Modelin kimliğini değiştiren fine-tuning yapıldıktan sonra 20 test sorusu ile eski kimlik vs yeni kimlik yanıt oranlarını sayısal olarak raporlama.
