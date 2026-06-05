# Laboratorio: Procesamiento de Lenguaje Natural
## Retos de investigación e integración de LLMs en Python

**Curso:** Inteligencia Artificial  
**Estudiante:** Legin-Pro  
**Repositorio:** https://github.com/Legin-Pro/lab-pln

---

## Descripción

Este laboratorio integra conceptos fundamentales de Procesamiento de Lenguaje Natural (PLN) con la API de Groq (LLM externo), aplicando 4 módulos prácticos que van desde preprocesamiento de texto hasta un chatbot conversacional.

---

## Proveedor LLM utilizado

| Proveedor | Modelo | SDK | Tier gratuito |
|-----------|--------|-----|---------------|
| **Groq** | llama-3.3-70b-versatile | `groq` (Python) | ✅ Sí |

**Por qué Groq:** Ofrece inferencia de alta velocidad con capa gratuita generosa, ideal para prototipos académicos.

---

## Estructura del proyecto

```
lab-pln/
├── laboratorio_pln.py    # Script principal con los 4 módulos
├── sentimientos.png      # Gráfica generada por el Módulo 3
└── README.md             # Este archivo
```

---

## Módulos implementados

### Módulo 1 — Preprocesamiento Híbrido
- **PLN local (NLTK):** tokenización, eliminación de stopwords, stemming con `SnowballStemmer`
- **Integración LLM (Groq):** extracción de sinónimos y antónimos en formato JSON estructurado
- **Prompt Engineering:** prompt zero-shot con instrucción de salida JSON estricta

### Módulo 2 — Clasificación y NER
- **PLN local (regex):** extracción de fechas y emails con expresiones regulares
- **Integración LLM (Groq):** clasificación spam/legítimo y NER (personas, organizaciones) con few-shot prompting
- **Dataset:** 5 correos simulados con casos reales de spam y comunicaciones legítimas

### Módulo 3 — Análisis de Sentimientos Temporal
- **PLN local:** asignación manual de scores por reglas léxicas
- **Integración LLM (Groq):** análisis de sentimiento con score -1.0 a 1.0 y fecha en formato JSON
- **Visualización:** gráfica matplotlib con línea de tendencia polinomial y zonas de color (positivo/negativo)

### Módulo 4 — Chatbot ARIA
- **Sistema:** chatbot conversacional con personalidad definida vía system prompt
- **Integración LLM (Groq):** historial de conversación multi-turno
- **Características:** modo texto interactivo, respuestas contextuales en español

---

## Instalación y uso

### Requisitos
```bash
pip install groq nltk matplotlib numpy
```

### Configuración
Editar la línea 2 de `laboratorio_pln.py` con tu API key de Groq:
```python
os.environ["GROQ_API_KEY"] = "tu_api_key_aqui"
```
Obtén tu key gratis en: https://console.groq.com

### Ejecución
```bash
python laboratorio_pln.py
```

El script ejecuta los 4 módulos en secuencia. El Módulo 3 genera una ventana con la gráfica (ciérrala para continuar). El Módulo 4 inicia el chatbot interactivo (escribe `salir` para terminar).

---

## Ejemplo de salida

**Módulo 1 — Sinónimos extraídos por LLM:**
```json
{
  "palabra": "inteligencia",
  "sinonimos": ["capacidad", "habilidad", "perspicacia"],
  "antonimos": ["ignorancia", "incapacidad", "estupidez"]
}
```

**Módulo 2 — Clasificación de correos:**
```json
{"id": 1, "tipo": "legítimo", "personas": ["María González"], "organizaciones": ["TechCorp"]}
{"id": 2, "tipo": "spam", "personas": [], "organizaciones": []}
```

**Módulo 3 — Gráfica de evolución de sentimiento:**

La gráfica muestra la evolución del sentimiento entre enero y junio 2024, con zonas rojas (negativo) y verdes (positivo), y una línea de tendencia proyectada.

---

## Conceptos aplicados

- Tokenización y stemming con NLTK
- Expresiones regulares para NER básico
- Zero-shot y few-shot prompting
- Análisis de sentimientos con LLMs
- Visualización de datos temporales con matplotlib
- Integración de APIs de LLMs con el SDK de Groq
- Diseño de system prompts para chatbots

---

## Recursos

- [Documentación Groq](https://console.groq.com/docs)
- [NLTK](https://www.nltk.org/)
- [Modelos disponibles en Groq](https://console.groq.com/docs/models)
