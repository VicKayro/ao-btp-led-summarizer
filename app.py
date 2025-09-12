import os
import re
import requests
import gradio as gr
import fitz  # donc PyMuPDF
from PIL import Image
import pytesseract
import io

# =========================
# Config
# =========================
MODEL_ID = "allenai/led-base-16384"
HF_TOKEN = os.getenv("HF_TOKEN")  # défini dans Settings -> Repository secrets
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
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
    """Appelle l'API Inference HF (LED 16k) en mode 'summarization'."""
    if not HF_TOKEN:
        return ("[ERREUR] Le secret HF_TOKEN est manquant.\n"
                "Dans ton Space : Settings → Repository secrets → ajoute 'HF_TOKEN' "
                "avec un token (scope: Read) créé sur https://huggingface.co/settings/tokens")
    payload = {
        "inputs": instructions + "\n\nDOCUMENT:\n" + text,
        "parameters": {
            "max_length": max_len,
            "min_length": min_len,
            "do_sample": False
        }
    }
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=180)
    if resp.status_code != 200:
        return f"[API {resp.status_code}] {resp.text}"
    data = resp.json()
    # Réponses typiques: [{"summary_text": "..."}] ou {"error": "..."} si modèle cold start
    if isinstance(data, dict) and "error" in data:
        return f"[API error] {data['error']}"
    try:
        return data[0]["summary_text"]
    except Exception:
        return str(data)

def extract_text_from_pdf(file_obj):
    """Essaie d'abord d'extraire le texte natif. 
    Si pas de texte → fallback OCR avec pytesseract."""
    doc = fitz.open(stream=file_obj.read(), filetype="pdf")
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
    """
    LED accepte ~16k tokens. On approxime en chars.
    On découpe par paragraphes pour éviter de couper en plein milieu.
    """
    paragraphs = t.split("\n\n")
    chunks, buf = [], ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= max_chars:
            buf += (("\n\n" if buf else "") + p)
        else:
            if buf:
                chunks.append(buf)
            if len(p) > max_chars:
                # Sécurité : coupe les très gros paragraphes
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars])
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks

def summarize_ao(pdf_file, objectif):
    # 1) Extraction
    raw = extract_text_from_pdf(pdf_file)
    text = clean_text(raw)
    if not text or len(text) < 200:
        return "Le PDF semble vide ou scanné (image). Ce POC ne fait pas d'OCR."

    instructions = build_instructions(objectif or DEFAULT_OBJECTIF)

    # 2) Chunking si le doc est long
    chunks = chunk_text_by_chars(text, max_chars=18000)

    # 3) Résumé par chunk
    partials = []
    for idx, chunk in enumerate(chunks, start=1):
        part = api_summarize(chunk, instructions, max_len=520, min_len=160)
        partials.append(f"### Résumé partiel {idx}\n{part}")

    # 4) Résumé final des résumés
    concat = "\n\n".join(partials)
    final = api_summarize(concat, instructions, max_len=700, min_len=200)
    return final

# =========================
# UI Gradio
# =========================
with gr.Blocks(theme=gr.themes.Soft(), title="Résumé d'AO BTP (LED 16k)") as demo:
    gr.Markdown("# 🏗️ Résumé d'AO BTP (LED 16k)\n"
                "Upload un AO (PDF). L’IA extrait **pièces à produire, livrables, contraintes**.\n"
                "*(Modèle : allenai/led-base-16384 via Hugging Face Inference API)*")

    with gr.Row():
        pdf = gr.File(label="PDF de l'appel d'offres", file_types=[".pdf"], type="binary")
        objectif = gr.Textbox(
            label="Objectif (facultatif)",
            value=DEFAULT_OBJECTIF,
            lines=4
        )

    run = gr.Button("Analyser le PDF")
    output = gr.Markdown(label="Résumé ciblé")

    def _run(pdf_file, obj):
        if not pdf_file:
            return "Merci d'uploader un PDF."
        return summarize_ao(pdf_file, obj)

    run.click(_run, inputs=[pdf, objectif], outputs=output)

demo.launch()
