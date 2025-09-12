---
title: Ao Btp Led Summarizer
emoji: 📉
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 5.45.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

---

# Résumé d'AO BTP (LED 16k)

Application Gradio hébergée sur Hugging Face Space.  
Lit des appels d'offres (PDF) et extrait :
- Pièces/documents à produire
- Livrables attendus
- Contraintes (délais, normes, certifications, formats)

**Modèle** : `allenai/led-base-16384` via Hugging Face Inference API.

## Lancer le Space
1. Crée un token HF (scope: Read) : https://huggingface.co/settings/tokens  
2. Dans le Space → Settings → Repository secrets → `HF_TOKEN` = ton token  
3. Fichiers requis : `app.py`, `requirements.txt`

> Remarque : Ce POC ne fait pas d’OCR (les PDF scannés ne sont pas pris en charge).

