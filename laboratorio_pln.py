import os, json, re
os.environ["GROQ_API_KEY"] = "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"

from groq import Groq
client = Groq(api_key=os.environ["GROQ_API_KEY"])

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

print("\n" + "="*60)
print("MÓDULO 1: PREPROCESAMIENTO HÍBRIDO")
print("="*60)

texto = """
La inteligencia artificial está transformando la educación.
Los estudiantes utilizan herramientas digitales para aprender
programación y análisis de datos en las universidades.
"""

# --- LOCAL: tokenización, stop words, stemming ---
tokens = [t for t in word_tokenize(texto.lower(), language='spanish') if t.isalpha()]
sin_stop = [t for t in tokens if t not in set(stopwords.words('spanish'))]
stems = [SnowballStemmer('spanish').stem(t) for t in sin_stop]

print(f"Tokens sin stop words: {sin_stop}")
print(f"Stemming: {stems}")

# --- API: palabras clave + sinónimos/antónimos ---
resp = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[{"role":"user","content":
        f"Del texto: '{' '.join(sin_stop)}'\n"
        f"Dame un JSON con: palabras_clave (lista de 5), y para cada una: sinonimos y antonimos.\n"
        f"Solo el JSON, sin texto extra."}],
    temperature=0.3
)
print("\nRespuesta API (sinónimos/antónimos):")
print(resp.choices[0].message.content)


print("\n" + "="*60)
print("MÓDULO 2: CLASIFICACIÓN Y NER")
print("="*60)

correos = [
    "Reunión el lunes 12 de junio a las 9am en TechCorp. Confirmar asistencia. - María González",
    "¡GANASTE $50,000! Haz clic AHORA en bit.ly/premio para reclamar tu dinero gratis.",
    "El sistema de la Universidad Nacional tendrá mantenimiento el viernes 15 de junio. - Dr. Roberto Sánchez",
    "Compra medicamentos SIN RECETA. Descuento 80% solo hoy. Contacta farmacia_secreta@gmail.com",
    "Adjunto informe Q2 de Inversiones del Sur S.A. Reunión el 20 de julio en Ciudad de México. - Ana Martínez",
]

# --- LOCAL: regex para fechas y emails ---
for i, c in enumerate(correos):
    fechas = re.findall(r'\d{1,2}\s+de\s+\w+', c)
    emails = re.findall(r'[\w.]+@[\w.]+', c)
    print(f"Correo {i+1} | Fechas: {fechas} | Emails: {emails}")

# --- API: clasificar spam + NER ---
resp2 = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[{"role":"user","content":
        f"Clasifica cada correo como spam o legítimo, y extrae personas y organizaciones.\n"
        f"Responde en JSON con lista: id, tipo, personas, organizaciones.\n"
        f"Correos:\n" + "\n".join([f"{i+1}. {c}" for i,c in enumerate(correos)])}],
    temperature=0.1
)
print("\nClasificación API:")
print(resp2.choices[0].message.content)


print("\n" + "="*60)
print("MÓDULO 3: SENTIMIENTOS TEMPORAL")
print("="*60)

opiniones = [
    ("2024-01-05", "Excelente producto, muy satisfecho con la compra."),
    ("2024-02-03", "Decepcionante, no coincide con la descripción."),
    ("2024-02-28", "Pésimo servicio, nadie responde los correos."),
    ("2024-03-22", "Regular, cumple su función pero nada especial."),
    ("2024-04-19", "Increíble calidad, superó todas mis expectativas."),
    ("2024-05-16", "Muy buen producto, llegó antes de lo esperado."),
    ("2024-06-01", "Fantástico, la empresa mejoró mucho este año."),
]

resp3 = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[{"role":"user","content":
        "Asigna un score de sentimiento de -1.0 a 1.0 a cada opinión.\n"
        "Responde SOLO un JSON: lista con fecha y score.\n"
        "Opiniones:\n" + "\n".join([f"{f}: {t}" for f,t in opiniones])}],
    temperature=0.1
)

print("Scores de sentimiento:")
contenido = resp3.choices[0].message.content
print(contenido)

# --- Graficar ---
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    import numpy as np

    data = json.loads(re.search(r'\[.*\]', contenido, re.DOTALL).group())
    fechas = [datetime.strptime(d['fecha'], "%Y-%m-%d") for d in data]
    scores = [d['score'] for d in data]

    x = mdates.date2num(fechas)
    z = np.polyfit(x, scores, 1)
    p = np.poly1d(z)

    plt.figure(figsize=(10,5))
    plt.plot(fechas, scores, 'o-', color='#2E5DA6', linewidth=2, label='Sentimiento')
    plt.plot(fechas, [p(xi) for xi in x], '--', color='purple', label='Tendencia')
    plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
    plt.fill_between(fechas, scores, 0, where=[s>0 for s in scores], alpha=0.1, color='green')
    plt.fill_between(fechas, scores, 0, where=[s<0 for s in scores], alpha=0.1, color='red')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    plt.title('Evolución del Sentimiento en el Tiempo')
    plt.ylabel('Score (-1 a 1)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('sentimientos.png', dpi=150)
    plt.show()
    print("Gráfica guardada como sentimientos.png")
except Exception as e:
    print(f"Instala matplotlib para la gráfica: pip install matplotlib numpy\nError: {e}")


print("\n" + "="*60)
print("MÓDULO 4: CHATBOT DE VOZ (modo texto)")
print("="*60)

historial = []
system = ("Eres ARIA, asistente virtual de tecnología. "
          "Responde en español, máximo 2 oraciones, tono amigable. "
          "No hables de política ni religión.")

print("Chatbot ARIA activo. Escribe 'salir' para terminar.\n")
while True:
    entrada = input("Tú: ").strip()
    if not entrada: continue
    if entrada.lower() == "salir":
        print("ARIA: ¡Hasta luego!")
        break

    historial.append({"role":"user","content":entrada})
    resp4 = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role":"system","content":system}] + historial,
        temperature=0.7, max_tokens=150
    )
    respuesta = resp4.choices[0].message.content
    historial.append({"role":"assistant","content":respuesta})
    print(f"ARIA: {respuesta}\n")

    # Síntesis de voz opcional
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.say(respuesta)
        engine.runAndWait()
    except:
        pass  # Si no tiene pyttsx3, solo muestra texto
