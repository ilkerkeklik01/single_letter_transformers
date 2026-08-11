"""Arac cagirabilen sohbet asistani (komut satiri).

Dongu cok basit:
    kullanici sorar -> model ya cevap verir ya da bir arac cagirir
                    -> araci biz calistirir, sonucu modele geri veririz
                    -> model nihai cevabi yazar

Kullanim:
    python3 chat.py
    python3 chat.py --embed-model magibu     # diger embedding modeliyle ara
"""

import argparse

import ollama_client
import tools

MAX_TOOL_ROUNDS = 5  # sonsuz arac dongusune karsi emniyet freni

SYSTEM_PROMPT = """Sen Turkce konusan yardimci bir asistansin. Elinde 4 arac var:

- medical_question : saglik/hastalik/belirti/tahlil/tedavi ile ilgili HER soru
- get_weather      : hava durumu
- get_exchange_rate: doviz kuru, para cevirme
- internet_search  : guncel olaylar ve genel bilgi

EN ONEMLI KURAL: medical_question aracinin dondurdugu metni AYNEN kullaniciya aktar.
Uzerine kendi tibbi bilgini ekleme, cevabi genisletme, duzeltme veya yorumlama.
Eger arac "Bilmiyorum" diyorsa sen de sadece bunu soyle; alternatif bir aciklama uretme.

Selamlasma ve sohbet gibi basit mesajlar icin arac cagirmana gerek yok."""


parser = argparse.ArgumentParser(description="Ollama tabanli arac cagirmali asistan.")
parser.add_argument(
    "--embed-model",
    default=ollama_client.DEFAULT_EMBED,
    choices=list(ollama_client.EMBED_MODELS),
    help="Tibbi aramada kullanilacak embedding modeli",
)
parser.add_argument("--chat-model", default=ollama_client.CHAT_MODEL, help="Ollama sohbet modeli")
args = parser.parse_args()

tools.ACTIVE_EMBED_KEY = args.embed_model


def run_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Modelin istedigi araclari calistirir ve sonuclari mesaj formatinda dondurur."""
    messages = []
    for call in tool_calls:
        name = call["function"]["name"]
        arguments = call["function"].get("arguments") or {}
        print(f"  🔧 {name}({arguments})")

        function = tools.TOOLS.get(name)
        if function is None:
            output = f"'{name}' adinda bir arac yok."
        else:
            try:
                output = function(**arguments)
            except Exception as exc:  # arac hatasi sohbeti bitirmesin
                output = f"Arac calistirilamadi: {exc}"

        messages.append({"role": "tool", "tool_name": name, "content": output})
    return messages


print("Ollama Egitim Asistani")
print(f"  sohbet modeli   : {args.chat_model}")
print(f"  embedding modeli: {ollama_client.EMBED_MODELS[args.embed_model]['name']}")
print("  cikmak icin: cik\n")

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

while True:
    try:
        question = input("Siz > ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not question:
        continue
    if question.lower() in {"cik", "çık", "exit", "quit"}:
        break

    messages.append({"role": "user", "content": question})

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            message = ollama_client.chat(
                messages, model=args.chat_model, tools=tools.TOOL_SCHEMAS
            )
            messages.append(message)
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                break
            messages.extend(run_tool_calls(tool_calls))
    except RuntimeError as exc:
        print(f"\nHata: {exc}\n")
        continue

    print(f"\nAsistan > {(message.get('content') or '').strip()}\n")
