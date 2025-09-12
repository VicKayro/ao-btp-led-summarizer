# Résumé d'Appels d'Offres BTP (LED 16k)

Application **Gradio** déployée sur Hugging Face Spaces.  
Elle permet d’analyser un **appel d’offres BTP (PDF)** et d’en extraire :

- Pièces techniques à produire  
- Livrables attendus  
- Contraintes majeures (délais, normes, certifications, formats)

## Stack technique
- [allenai/led-large-16384](https://huggingface.co/allenai/led-large-16384) pour lire des documents longs (16k tokens)
- Fallback automatique sur [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) si LED est indispo
- OCR via PyMuPDF + Tesseract pour gérer les PDF scannés
- Gradio pour l’UI

## Installation locale
```bash
git clone https://github.com/<ton-user>/ao-btp-led-summarizer.git
cd ao-btp-led-summarizer
pip install -r requirements.txt
python app.py
