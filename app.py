import os
import re
import requests
import gradio as gr
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io

# =========================
# Config
# =========================
PRIMARY_MODEL = "allenai/led-large-16384"
FALLBACK_MODEL = "facebook/bart-large-cnn"
HF_TOKEN = os.getenv("HF_TOKEN")  # défini dans Settings -> Repository secrets
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

DEFAULT_OBJECTIF = (
    "1) Pièces et documents techniques à produire "
    "2) Livrables attendus "
    "3) Contraintes majeures (délais, normes, certifications, formats)."
)

# =========================
# Utils
# =========================
def build_instructions(objectif: str) -> str:
    return f"""Tu es un assistant expert en marchés publics dans le BTP.
Lis l'extrait d'appel d'offres ci-dessous et produis un résumé ciblé, concis et actionnable.
Ne fais PAS un résumé narratif global, mais un extrait orienté exécution.
Objectif utilisateur :
{objectif}
Contraintes :
- Sois exhaustif sur les éléments demandés dans l'objectif, pas le reste.
- Utilise des puces claires (une idée = une puce).
- Normalise les intitulés quand c'est possible (ex: DOE = Dossier des Ouvrages Exécutés).
- Si une info est absente, écris "Non précisé".
FORMAT DE SORTIE STRICT (Markdown) :
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

def api_summarize(text: str, instructions: str, max_len=520, min_len=160) -> str:
    if not HF_TOKEN:
        return ("[ERREUR] Le secret HF_TOKEN est manquant.\n"
                "➡️ Va dans ton Space : Settings → Repository secrets → ajoute 'HF_TOKEN'\n"
                "avec un token Hugging Face (scope: Read).")

    model = PRIMARY_MODEL
    for attempt in range(2):  # essaye 2 fois : LED puis fallback
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        payload = {
            "inputs": instructions + "\n\nDOCUMENT:\n" + text,
            "parameters": {"max_length": max_len, "min_length": min_len, "do_sample": False}
        }
        resp = requests.post(api_url, headers=HEADERS, json=payload, timeout=180)

        if resp.status_code == 200:
            try:
                data = resp.json()
                return data[0]["summary_text"]
            except Exception:
                return str(data)
        else:
            if attempt == 0:
                print(f"⚠️ LED indisponible ({resp.status_code}), fallback sur BART")
                model = FALLBACK_MODEL
            else:
                return f"[API {resp.status_code}] {resp.text}"

def extract_text_from_pdf(file_obj):
    """Essaie d'abord d'extraire le texte natif. 
    Si pas de texte → fallback OCR avec pytesseract."""
    doc = fitz.open(stream=file_obj, filetype="pdf")
    texts = []
    for page in doc:
        t = page.get_text("text")
        if not t.strip():  
            # pas de texte -> OCR
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            t = pytesseract.image_to_string(img, lang="fra")  # OCR FR
        texts.append(t)
    return "\n".join(texts)

def clean_text(t: str) -> str:
    t = re.sub(r"\r", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()

def chunk_text_by_chars(t: str, max_chars: int = 18000):
    """Découpe en chunks ~16k tokens (approx chars)."""
    paragraphs = t.split("\n\n")
    chunks, buf = [], ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= max_chars:
            buf += (("\n\n" if buf else "") + p)
        else:
            if buf:
                chunks.append(buf)
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars])
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks

# =========================
# UI Gradio
# =========================
with gr.Blocks(theme=gr.themes.Soft(), title="Résumé d'AO BTP (LED 16k)") as demo:
    gr.Markdown("# 🏗️ Résumé d'AO BTP (LED 16k)\n"
                "Upload un AO (PDF). L’IA extrait **pièces à produire, livrables, contraintes**.\n"
                "*(Modèle : allenai/led-large-16384 via Hugging Face Inference API, fallback BART si indispo)*")

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
    
        instructions = build_instructions(obj or DEFAULT_OBJECTIF)
        chunks = chunk_text_by_chars(text, max_chars=18000)
    
        logs_text = "⏳ Début de l'analyse...\n"
        yield 0, logs_text, ""  # reset tout
    
        partials = []
        total = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            part = api_summarize(chunk, instructions, max_len=520, min_len=160)
            partials.append(part)
    
            logs_text += f"- Chunk {idx}/{total} analysé ✅\n"
            progress_value = int((idx / total) * 80)
            yield progress_value, logs_text, ""
    
        logs_text += "\n⏳ Génération du résumé final..."
        yield 90, logs_text, ""
    
        concat = "\n\n".join(partials)
        final = api_summarize(concat, instructions, max_len=700, min_len=200)
    
        logs_text += "\n✅ Résumé final généré."
        yield 100, logs_text, final

    run.click(_run, inputs=[pdf, objectif], outputs=[progress, logs, output])

demo.launch()
