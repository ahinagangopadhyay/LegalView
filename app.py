from flask import Flask, request, jsonify, render_template
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import PyPDF2
import spacy
import os
import zipfile
import gdown

app = Flask(__name__)

# --- Model Configuration ---
model_zip = "model.zip"
model_extract_path = "model"  # unzip to this folder
model_dir = "model/bert_risk_model"  # actual path to model files
file_id = "1hElMtNrmZp4d4k9s_5qXSUzFstzCC2lt"

# --- Step 1: Download + Extract if not already ---
def setup_model():
    if not os.path.exists(model_dir):
        print("[INFO] Downloading model from Google Drive...")
        gdown.download(id=file_id, output=model_zip, quiet=False)

        print("[INFO] Extracting model.zip...")
        with zipfile.ZipFile(model_zip, 'r') as zip_ref:
            zip_ref.extractall(model_extract_path)

        print("[INFO] Model ready.")
    else:
        print("[INFO] Model already exists.")

setup_model()

# --- Step 2: Load NLP and Models ---
nlp = spacy.load("en_core_web_sm")

tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
model = DistilBertForSequenceClassification.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

# Classify sentence as risky/safe
def classify_clause(text):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    encoded.pop("token_type_ids", None)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return pred

# Extract text from PDF
def extract_text_from_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in pdf.pages)

# Summarize long text
def summarize_text(text, max_chunk_len=1024):
    chunks = [text[i:i+max_chunk_len] for i in range(0, len(text), max_chunk_len)]
    summaries = [
        summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        for chunk in chunks
    ]
    return "\n".join(summaries)

# Main analysis endpoint
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' in request.files:
        file = request.files['file']
        text = extract_text_from_pdf(file)
    else:
        data = request.get_json()
        text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    results = []
    risky = []

    for sent in sentences:
        label = classify_clause(sent)
        results.append({"sentence": sent, "risk": "RISKY" if label == 1 else "SAFE"})
        if label == 1:
            risky.append(sent)

    summary = summarize_text(text)

    return jsonify({
        "summary": summary,
        "classified_sentences": results,
        "risky_clauses_only": risky
    })

# --- Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
