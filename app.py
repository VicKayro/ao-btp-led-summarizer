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
PRIMARY_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
FALLBACK_MODEL = "facebook/bart-large-cnn"
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

DEFAULT_OBJECTIF = (
    "1) Pièces et documents techniques à produire "
    "2) Livrables attendus "
    "3) Contraintes majeures (délais, normes, certifications, formats)."
)

# =========================
# Utils
# =========================
def build_prompt(objectif: str, text: str) -> str:
    return f"""Tu es un assistant expert en marchés publics BTP. 
Lis l'appel d'offres suivant et produis un résumé **orienté exécution**. 
Ne fais PAS de résumé narratif global.

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

def api_infer(model: str, prompt: str, max_tokens=700):
    if not HF_TOKEN:
        return ("[ERREUR] Le secret HF_TOKEN est manquant.\n"
                "➡️ Va dans ton Space : Settings → Repository secrets → ajoute 'HF_TOKEN'\n"
                "avec un token Hugging Face (scope: Read).")

    url = f"https://api-inference.huggingface.co/models/{model}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.2,
            "return_full_text": False
        }
    }
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=180)
    if resp.status_code != 200:
        return f"[API {resp.status_code}] {resp.text}"
    try:
        data = resp.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        else:
            return str(data)
    except Exception:
        return str(resp.text)

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

def chunk_text_by_chars(t: str, max_chars: int = 8000):
    """On limite à ~8k caractères par chunk (Mistral-7B supporte ~32k tokens)."""
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
with gr.Blocks(theme=gr.themes.Soft(), title="Résumé d'AO BTP (Mistral)") as demo:
    gr.Markdown("# 🏗️ Résumé d'AO BTP (Mistral Instruct)\n"
                "Upload un AO (PDF). L’IA extrait **pièces à produire, livrables, contraintes**.\n"
                "*(Modèle : Mistral-7B-Instruct via Hugging Face Inference API, fallback BART si indispo)*")

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
        chunks = chunk_text_by_chars(text, max_chars=8000)
    
        logs_text = "⏳ Début de l'analyse...\n"
        yield 0, logs_text, ""
    
        partials = []
        total = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            prompt = build_prompt(objectif_final, chunk)
            part = api_infer(PRIMARY_MODEL, prompt, max_tokens=500)
            partials.append(part)
    
            logs_text += f"- Chunk {idx}/{total} analysé ✅\n"
            progress_value = int((idx / total) * 80)
            yield progress_value, logs_text, ""
    
        logs_text += "\n⏳ Génération du résumé final..."
        yield 90, logs_text, ""
    
        concat = "\n\n".join(partials)
        prompt_final = build_prompt(objectif_final, concat)
        final = api_infer(PRIMARY_MODEL, prompt_final, max_tokens=700)
    
        logs_text += "\n✅ Résumé final généré."
        yield 100, logs_text, final

    run.click(_run, inputs=[pdf, objectif], outputs=[progress, logs, output])

demo.launch()
