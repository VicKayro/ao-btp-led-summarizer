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

# 🏗Résumé d'Appels d'Offres (GPT-4o-mini)

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange?logo=gradio)](https://gradio.app/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green?logo=openai)](https://openai.com/)
[![Hugging Face](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces)

---

## 📖 Description

Cette application lit des **appels d’offres (PDF)** dans le domaine du **BTP** et génère un résumé **structuré et actionnable** :

- **Pièces et documents techniques à produire**  
- **Livrables attendus**  
- ⚠**Contraintes majeures** (délais, normes, certifications, formats)  
- **Points ambigus / à clarifier**  

Elle utilise **GPT-4o-mini (OpenAI API)** avec chunking intelligent → capable de traiter des documents longs (jusqu’à ~100 pages).

---

## Capture d’écran

<img width="1175" height="464" alt="image" src="https://github.com/user-attachments/assets/2614b716-447f-4a2f-9eb0-49f1867b17f2" />


---

## ⚙Stack technique

- [Gradio](https://gradio.app/) → Interface utilisateur simple et élégante  
- [PyMuPDF](https://pymupdf.readthedocs.io/) → Extraction de texte natif des PDF  
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) → Reconnaissance de texte sur PDF scannés  
- [OpenAI GPT-4o-mini](https://platform.openai.com/) → Analyse et résumé structuré avec contexte long  
- [Hugging Face Spaces](https://huggingface.co/spaces) → Déploiement serverless gratuit  

---

## 🛠Installation locale

### 1. Clone le projet
```bash
git clone https://github.com/<ton-user>/ao-btp-led-summarizer.git
cd ao-btp-led-summarizer
````

### 2. Installe les dépendances

```bash
pip install -r requirements.txt
```

### 3. Configure la clé OpenAI

Crée un fichier `.env` ou exporte ta clé dans le terminal :

```bash
export OPENAI_API_KEY="ta_cle_api_openai"
```

### 4. Lance l’app

```bash
python app.py
```

---

## Déploiement sur Hugging Face Spaces

1. Crée un Space (SDK = Gradio, Runtime = Python 3.10)
2. Pousse `app.py`, `requirements.txt`, `README.md`, et une `screenshot.png`
3. Dans **Settings → Repository secrets**, ajoute :

   * `OPENAI_API_KEY` = ta clé OpenAI
4. Redémarre le Space → ton app est en ligne 🎉

---

## Exemple d’usage

* **Avant** : un appel d’offres BTP de 100 pages = lecture manuelle fastidieuse
* **Avec cette app** : en quelques minutes → tu obtiens directement :

  * Les **pièces à fournir** (ex: DOE, plans, notices techniques)
  * Les **livrables attendus**
  * Les **contraintes** (normes NF, certifications ISO, délais…)

Un vrai **assistant de pré-analyse AO** pour les BE / services marchés.

---

## Licence

MIT – utilisation libre et open source.
N’hésitez pas à contribuer (pull requests bienvenues 🚀).

```
