"""Cosine benzerliğinin ölçekten değil yönden etkilendiği üç basit örnek."""

from rag_lab.similarity import cosine_similarity, dot, l2_norm, normalize


pairs = [
    ([1, 0], [10, 0], "aynı yön, farklı uzunluk"),
    ([1, 0], [0, 3], "dik yönler"),
    ([1, 1], [-1, -1], "zıt yönler"),
]

for a, b, label in pairs:
    score = cosine_similarity(a, b)
    print(f"{label:28} a={a}, b={b}, cosine={score:+.3f}")

a, b = [1, 1], [2, 0]
print("\nElle hesap:")
print(f"dot={dot(a, b)}, ||a||={l2_norm(a):.3f}, ||b||={l2_norm(b):.3f}")
print(f"cosine = dot / (||a||·||b||) = {cosine_similarity(a, b):.3f}")
print(f"normalize(a)={normalize(a)}")

try:
    cosine_similarity([0, 0], [1, 0])
except ValueError as error:
    print(f"\nSıfır vektör kontrolü: {error}")
