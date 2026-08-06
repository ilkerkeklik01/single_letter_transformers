"""Aynı metni üç stratejiyle parçalar ve sınırların nasıl değiştiğini gösterir."""

from rag_lab.chunking import fixed_word_chunks, markdown_chunks, sentence_chunks


TEXT = """# Kurulum
Uygulamayı indirin. Hesabınızla giriş yapın. İlk eşitleme birkaç dakika sürebilir.

# Sorun giderme
Bağlantı kurulmazsa ağı kontrol edin. Ardından uygulamayı yeniden başlatın.
Sorun sürerse destek ekibine hata koduyla ulaşın."""

strategies = {
    "sabit kelime (size=10, overlap=2)": fixed_word_chunks(TEXT, size=10, overlap=2),
    "cümle (max_words=12, overlap=1)": sentence_chunks(TEXT, max_words=12, overlap_sentences=1),
    "Markdown başlığı (max_words=14)": markdown_chunks(TEXT, max_words=14, overlap=2),
}

for name, chunks in strategies.items():
    print(f"\n{name}\n{'-' * len(name)}")
    for index, chunk in enumerate(chunks):
        print(f"[{index}] {chunk!r}")
