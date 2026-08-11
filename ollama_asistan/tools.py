"""Modelin cagirabilecegi 4 arac.

Her arac sade bir Python fonksiyonudur ve METIN dondurur; bu metin modele
geri beslenir. Hicbiri hata firlatmaz — hata olursa Turkce bir aciklama doner,
boylece sohbet dongusu cokmez.

TOOL_SCHEMAS listesi ise modele "elinde su araclar var" demenin JSON halidir.
"""

import html
import re

import requests

import medical_rag

TIMEOUT = 20
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# WMO hava durumu kodlarindan Turkce aciklamaya (Open-Meteo bu kodlari kullanir).
WMO_TR = {
    0: "acik", 1: "az bulutlu", 2: "parcali bulutlu", 3: "cok bulutlu",
    45: "sisli", 48: "kirragi sisi", 51: "hafif ciseleme", 53: "ciseleme",
    55: "yogun ciseleme", 61: "hafif yagmur", 63: "yagmurlu", 65: "kuvvetli yagmur",
    71: "hafif kar", 73: "kar yagisli", 75: "yogun kar", 77: "kar taneli",
    80: "saganak", 81: "kuvvetli saganak", 82: "siddetli saganak",
    85: "kar saganagi", 86: "yogun kar saganagi",
    95: "gok gurultulu firtina", 96: "dolulu firtina", 99: "siddetli dolulu firtina",
}

# chat.py bunu ayarlar: hangi embedding modeliyle arama yapilacagi.
ACTIVE_EMBED_KEY = medical_rag.ollama_client.DEFAULT_EMBED


def internet_search(query: str, max_results: int = 5) -> str:
    """DuckDuckGo'nun sade (lite) arayuzunde arama yapar. API anahtari gerekmez."""
    try:
        response = requests.post(
            "https://lite.duckduckgo.com/lite/",
            data={"q": query},
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        # DDG'nin HTML'inde href, class'tan once gelir ve class tek tirnaklidir:
        #   <a rel="nofollow" href="https://..." class='result-link'>Baslik</a>
        pairs = re.findall(
            r"""<a[^>]*href="([^"]+)"[^>]*class=['"]result-link['"][^>]*>(.*?)</a>""",
            response.text,
            flags=re.DOTALL,
        )
        results = []
        for url, raw_title in pairs[:max_results]:
            title = html.unescape(re.sub(r"<[^>]+>", "", raw_title)).strip()
            if title:
                results.append(f"{len(results) + 1}. {title}\n   {html.unescape(url)}")
        if results:
            return f"'{query}' icin internet sonuclari:\n" + "\n".join(results)
    except requests.RequestException:
        pass  # asagidaki Wikipedia yedegine dus

    return _wikipedia_search(query, max_results)


def _wikipedia_search(query: str, max_results: int) -> str:
    """Yedek arama: Turkce Wikipedia API'si."""
    try:
        data = requests.get(
            "https://tr.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search", "srsearch": query,
                "srlimit": max_results, "format": "json",
            },
            headers=HEADERS,
            timeout=TIMEOUT,
        ).json()
        items = data.get("query", {}).get("search", [])
        if not items:
            return f"'{query}' icin sonuc bulunamadi."
        lines = []
        for i, item in enumerate(items, start=1):
            snippet = html.unescape(re.sub(r"<[^>]+>", "", item.get("snippet", "")))
            slug = item["title"].replace(" ", "_")
            lines.append(f"{i}. {item['title']}\n   {snippet}\n   https://tr.wikipedia.org/wiki/{slug}")
        return f"'{query}' icin Wikipedia sonuclari:\n" + "\n".join(lines)
    except requests.RequestException as exc:
        return f"Arama yapilamadi: {exc}"


def get_weather(city: str) -> str:
    """Bir sehrin guncel hava durumunu Open-Meteo'dan getirir. API anahtari gerekmez."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "tr"},
            timeout=TIMEOUT,
        ).json()
        places = geo.get("results")
        if not places:
            return f"'{city}' adinda bir sehir bulunamadi."
        place = places[0]

        data = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "timezone": "auto",
            },
            timeout=TIMEOUT,
        ).json()
        now = data["current"]
        description = WMO_TR.get(now["weather_code"], "bilinmiyor")
        return (
            f"{place['name']} ({place.get('country', '')}) hava durumu: {description}, "
            f"{now['temperature_2m']}°C, nem %{now['relative_humidity_2m']}, "
            f"ruzgar {now['wind_speed_10m']} km/s. (Olcum: {now['time']})"
        )
    except (requests.RequestException, KeyError) as exc:
        return f"Hava durumu alinamadi: {exc}"


def get_exchange_rate(from_currency: str, to_currency: str, amount: float = 1.0) -> str:
    """Guncel doviz kurunu Frankfurter API'sinden getirir. API anahtari gerekmez."""
    source = from_currency.strip().upper()
    target = to_currency.strip().upper()
    try:
        data = requests.get(
            "https://api.frankfurter.dev/v1/latest",
            params={"base": source, "symbols": target},
            timeout=TIMEOUT,
        ).json()
        rate = data.get("rates", {}).get(target)
        if rate is None:
            return f"{source} -> {target} kuru bulunamadi. Para birimi kodlarini kontrol edin."
        return (
            f"1 {source} = {rate} {target} ({data['date']} tarihli kur). "
            f"{amount} {source} = {round(float(amount) * rate, 2)} {target}"
        )
    except (requests.RequestException, ValueError) as exc:
        return f"Kur bilgisi alinamadi: {exc}"


def medical_question(question: str) -> str:
    """Tibbi soruyu SADECE indekslenmis makalelere dayanarak cevaplar.

    Cevap burada bitmis haldedir; chat.py'deki model bunu aynen aktarmalidir.
    """
    result = medical_rag.answer_medical(question, embed_key=ACTIVE_EMBED_KEY)
    if not result["grounded"]:
        return result["answer"]
    sources = "\n".join(
        f"- {s['title']} (benzerlik {s['similarity']})\n  {s['url']}" for s in result["sources"]
    )
    return f"{result['answer']}\n\nKaynaklar:\n{sources}"


TOOLS = {
    "internet_search": internet_search,
    "get_weather": get_weather,
    "get_exchange_rate": get_exchange_rate,
    "medical_question": medical_question,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "internet_search",
            "description": (
                "Guncel olaylar, haberler veya genel bilgi icin internette arama yapar. "
                "Tibbi sorular icin KULLANMA."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Arama sorgusu"},
                    "max_results": {"type": "integer", "description": "Sonuc sayisi (varsayilan 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Bir sehrin guncel hava durumunu (sicaklik, nem, ruzgar) getirir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Sehir adi, ornegin 'Istanbul'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Iki para birimi arasindaki guncel kuru ve cevrilmis tutari getirir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {"type": "string", "description": "Kaynak para birimi kodu, ornegin USD"},
                    "to_currency": {"type": "string", "description": "Hedef para birimi kodu, ornegin TRY"},
                    "amount": {"type": "number", "description": "Cevrilecek tutar (varsayilan 1)"},
                },
                "required": ["from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "medical_question",
            "description": (
                "SAGLIK, hastalik, belirti, tahlil, tedavi veya ilac ile ilgili TUM sorular icin "
                "bu araci kullan. Cevabi hastane makaleleri veritabanindan uretir. "
                "Donen metni aynen kullanicaya aktar, uzerine kendi bilgini EKLEME."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Kullanicinin tibbi sorusu"},
                },
                "required": ["question"],
            },
        },
    },
]
