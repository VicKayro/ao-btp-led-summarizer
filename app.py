import os
import re
import gradio as gr
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
from openai import OpenAI

# =========================
# Config
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # défini dans Settings -> Repository secrets
client = OpenAI(api_key=OPENAI_API_KEY)

DEFAULT_OBJECTIF = (
    "1) Pièces et documents techniques à produire "
    "2) Livrables attendus "
    "3) Contraintes majeures (délais, normes, certifications, formats)."
)

# =========================
# Utils
# =========================
def build_prompt(objectif: str, text: str) -> str:
    return f"""Tu es un assistant expert en analise d'appels d'offrs, et en préparation de réponse. 
Analyse l'appel d'offres ci-dessous et produis un résumé **orienté exécution**.

Objectif :
{objectif}

Texte :
{text}

Format de sortie attendu (Markdown) :
# Résumé ciblé AO BTP
## 📦 Pièces à produire
- ...
## 📄 Livrables attendus
- ...
## ⚠️ Contraintes majeures
- Délais : ...
- Normes : ...
- Certifications : ...
- Formats / Modalités de rendu : ...
## 🔎 Points ambigus / À clarifier
- ...
"""

def openai_summarize(prompt: str, max_tokens=800) -> str:
    if not OPENAI_API_KEY:
        return "[ERREUR] Le secret OPENAI_API_KEY est manquant. Ajoute-le dans Settings → Repository secrets."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # rapide, contexte long (~128k)
            messages=[
                {"role": "system", "content": "Tu es un assistant expert en appels d'offres."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.2
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[OpenAI ERROR] {e}"

def extract_text_from_pdf(file_obj):
    """Texte natif sinon OCR avec pytesseract."""
    doc = fitz.open(stream=file_obj, filetype="pdf")
    texts = []
    for page in doc:
        t = page.get_text("text")
        if not t.strip():
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            t = pytesseract.image_to_string(img, lang="fra")
        texts.append(t)
    return "\n".join(texts)

def clean_text(t: str) -> str:
    t = re.sub(r"\r", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()

def chunk_text_by_chars(t: str, max_chars: int = 15000):
    """Découpe en chunks pour GPT-4o-mini (contexte long)."""
    paragraphs = t.split("\n\n")
    chunks, buf = [], ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= max_chars:
            buf += (("\n\n" if buf else "") + p)
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks

# =========================
# UI Gradio
# =========================
with gr.Blocks(theme=gr.themes.Soft(), title="Résumé d'AO (GPT-4o-mini)") as demo:
    gr.Markdown("# 🏗️ Résumé d'AO (GPT-4o-mini)\n"
                "Upload un AO (PDF). L’IA extrait **pièces à produire, livrables, contraintes**.\n"
                "*(Modèle : GPT-4o-mini via OpenAI API, contexte long)*")

    with gr.Row():
        pdf = gr.File(label="PDF de l'appel d'offres", file_types=[".pdf"], type="binary")
        objectif = gr.Textbox(
            label="Objectif (facultatif)",
            value=DEFAULT_OBJECTIF,
            lines=4
        )

    run = gr.Button("Analyser le PDF", variant="primary")
    progress = gr.Slider(label="Progression", minimum=0, maximum=100, value=0, step=1, interactive=False)
    logs = gr.Markdown(label="Logs", value="Aucune analyse lancée.")
    output = gr.Markdown(label="Résumé ciblé")

    def _run(pdf_file, obj):
        if not pdf_file:
            yield 0, "Merci d'uploader un PDF.", ""
            return

        raw = extract_text_from_pdf(pdf_file)
        text = clean_text(raw)
        if not text or len(text) < 200:
            yield 0, "⚠️ PDF vide ou scanné sans OCR utilisable.", ""
            return

        objectif_final = obj or DEFAULT_OBJECTIF
        chunks = chunk_text_by_chars(text, max_chars=15000)

        logs_text = "⏳ Début de l'analyse...\n"
        yield 0, logs_text, ""

        partials = []
        total = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            prompt = build_prompt(objectif_final, chunk)
            part = openai_summarize(prompt, max_tokens=600)
            partials.append(part)

            logs_text += f"- Chunk {idx}/{total} analysé ✅\n"
            progress_value = int((idx / total) * 80)
            yield progress_value, logs_text, ""

        logs_text += "\n⏳ Génération du résumé final..."
        yield 90, logs_text, ""

        concat = "\n\n".join(partials)
        prompt_final = build_prompt(objectif_final, concat)
        final = openai_summarize(prompt_final, max_tokens=1000)

        logs_text += "\n✅ Résumé final généré."
        yield 100, logs_text, final

    run.click(_run, inputs=[pdf, objectif], outputs=[progress, logs, output])

demo.launch()
