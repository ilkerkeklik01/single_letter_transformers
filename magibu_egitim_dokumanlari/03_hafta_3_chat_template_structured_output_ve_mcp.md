# Hafta 3: Chat Templates, Structured Output ve Model Context Protocol (MCP)

Bu doküman, Magibu Akademi Uygulamalı Yapay Zekâ Mimarisi Eğitimi'nin 3. haftasında işlenen tüm teknik konuları, Jinja2 şablonlarını, JSON kısıtlı örnekleme (constrained sampling) metotlarını ve saf Python ile Agent Loop (ReAct) kodlama adımlarını içermektedir.

---

## 📺 İlgili Video Kaynakları
- 🎥 [Magibu Uygulamalı Yapay Zekâ Mimarisi | Hafta 3, Gün 1: LLM'lerde Tool Calling ve Structured Output](https://www.youtube.com/watch?v=nlqbUgSECm8)
- 🎥 [Magibu Uygulamalı Yapay Zekâ Mimarisi | Hafta 3, Gün 2: Chat Template ve Structured Output](https://www.youtube.com/watch?v=K7lFadCfFL8)

---

## 1. Chat Template (Sohbet Şablonları) ve Jinja2

İki veya daha fazla tur sürdürülen diyaloglarda (`user`, `assistant`, `system`, `tool`), mesaj listesinin dil modelinin beklediği özel kontrol token'larına dökülmesini sağlayan Jinja2 kodlarına **Chat Template** denir.

### Jinja2 Şablon Yapısı (Örnek `tokenizer_config.json`):
```jinja2
{% for message in messages %}
  {% if message['role'] == 'user' %}
    {{ '<|im_start|>user
' + message['content'] + '<|im_end|>
' }}
  {% elif message['role'] == 'assistant' %}
    {{ '<|im_start|>assistant
' + message['content'] + '<|im_end|>
' }}
  {% endif %}
{% endfor %}
{% if add_generation_prompt %}
  {{ '<|im_start|>assistant
' }}
{% endif %}
```

- **`add_generation_prompt=True`:** Dizinin sonuna `<|im_start|>assistant
` ekleyerek sıranın asisatana geçtiğini modele bildirir ve yanıt üretmesini başlatır.
- **Thinking Manipülasyonu:** Düşünme adımı olan modellerde (`<think>...</think>`), modele üretilen yanıtın hemen başına kapama etiketini (`</think>`) enjekte ederek düşünme adımını zorla atlatmak mümkündür.

---

## 2. Yapılandırılmış Çıktı (Structured Output) & Constrained Sampling

LLM'in çıktı olarak serbest metin yerine %100 geçerli JSON veya Pydantic şemasına uyan veri üretmesini sağlama yöntemleri:

1. **Prompt Engineering:** Promp'ta JSON şeması vermek (Garantisi yoktur, model halüsinasyon görebilir).
2. **Grammar / Constrained Sampling (Ollama, Outlines, vLLM, Guidance):** Softmax katmanında Go/C++ dilinde derlenmiş gramer kuralları ile yalnızca JSON şemasına uyan token'ların olasılığını açık tutarak %100 geçerli JSON çıktısı alma.

```python
import pydantic
from pydantic import BaseModel

class WeatherRequest(BaseModel):
    city: str
    unit: str = "celsius"

# Pydantic şemasının JSON Schema karşılığı modele sistem komutunda verilir
json_schema = WeatherRequest.model_json_schema()
```

---

## 3. Tool Calling & Agentic Loop (ReAct Mimarisi)

Model doğrudan dış API'ları çalıştırmaz; çalıştırmak istediği fonksiyonun adını ve argümanlarını JSON biçiminde üretir. Uygulama (*harness / agent loop*) fonksiyonu çalıştırıp sonucu tekrar modele besler.

### Saf Python ile Agent Loop Uygulama Mimarisi:

```python
import ollama

def get_weather(city: str) -> str:
    # Gerçek veya simüle edilmiş API çağrısı
    return f"{city} için hava sıcaklığı 31°C ve güneşlidir."

tools_schema = [{
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Belirtilen şehrin hava durumunu getirir.',
        'parameters': {
            'type': 'object',
            'properties': {'city': {'type': 'string', 'description': 'Şehir adı'}},
            'required': ['city']
        }
    }
}]

messages = [{'role': 'user', 'content': 'İstanbul`da hava bugün nasıl?'}]

# 1. Döngü: Model Tool Call talep eder
response = ollama.chat(model='gemma:12b', messages=messages, tools=tools_schema)

if response.message.tool_calls:
    for tool_call in response.message.tool_calls:
        func_name = tool_call.function.name
        args = tool_call.function.arguments
        
        # 2. Fonksiyon Çalıştırma
        if func_name == 'get_weather':
            result = get_weather(args['city'])
            
            # 3. Sonucu Modele Geri Besleme
            messages.append(response.message)
            messages.append({'role': 'tool', 'name': func_name, 'content': result})

# 4. Final Yanıtı Alma
final_response = ollama.chat(model='gemma:12b', messages=messages)
print(final_response.message.content)
```

---

## 4. Hafta 3 Ödevi ve Hugging Face Space Dağıtımı

1. **Özel Hikaye & Dış API Senaryosu:**
   - E-Ticaret, kafe sipariş, uçak bileti veya kütüphane sistemi için 3+ fonksiyon içeren Tool Calling ajanı yazma.
2. **Hugging Face Space Yayımlama:**
   - Streamlit veya Gradio ile UI arayüzü yazıp Hugging Face Space üzerinde canlıya alma.
   - Modele yapılan Tool Call turlarını ve ham JSON alışverişlerini ekranda gösteren geçiş paneli (*Inspector UI*) ekleme.
