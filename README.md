# 📜 LegalView - AI Legal Clause Summarizer

**LegalView** is an AI-powered web app that simplifies complex legal documents like contracts, privacy policies, and rental agreements. It generates concise, bullet-point summaries and highlights risky or unusual clauses to make legal content more transparent and understandable for everyone.

---

## 🚀 Features

- 🔍 **Clause Detection & Summarization**
  - Uses NLP and ML to extract and simplify key clauses.
  
- ⚠️ **Risk Flagging**
  - Flags potentially risky, harmful, or unusual clauses for user awareness.

- 📄 **Supports Common Legal Docs**
  - Contracts, Privacy Policies, Rental Agreements, Terms of Service.

- 🌐 **User-Friendly Web Interface**
  - Built using Flask and deployed on Hugging Face Spaces with Docker support.

---

## 🧠 Tech Stack

- **Frontend:** HTML, CSS, Bootstrap (optional React UI upcoming)
- **Backend:** Python, Flask
- **ML/NLP:** Scikit-learn, Pandas, TfidfVectorizer, Logistic Regression, spaCy, HuggingFace Transformers (DistilBERT + BART)
- **Deployment:** Hugging Face Spaces (with Docker)

---

## 📂 Folder Structure

```
LegalView/
├── app.py                 # Main Flask application
├── templates/             # HTML templates
│   └── index.html
├── static/                # CSS and JS assets
├── requirements.txt       # Python dependencies
├── Dockerfile             # For Hugging Face deployment
└── README.md              # Project overview
```

---

## 📦 How to Run Locally

```bash
# 1. Clone the repository
https://github.com/ahinagangopadhyay/LegalView.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Flask app
python app.py
```

> 📁 Note: The ML model (`model.pkl`) and tokenizer (`tokenizer.pkl`) are downloaded automatically from Google Drive during runtime for portability. The model was trained in the provided Jupyter notebook.

---

## 📊 Project Performance

- Achieved **95.6% accuracy**, **0.78 F1 score**, and **0.86 precision** on a labeled legal clause dataset.
- Fine-tuned **DistilBERT** for clause classification (Safe/Unsafe)
- Used **BART** for abstractive summarization
- Supported both PDF upload and raw text input
- Used **spaCy** for legal sentence segmentation

---

## 🌍 Live Deployment
👉 [Visit LegalView on Hugging Face Spaces](https://huggingface.co/spaces/ahinaganguly/legalview)

---

## 🙌 Credits
Built with ❤️ by [Ahina](https://github.com/ahinagangopadhyay)

---

