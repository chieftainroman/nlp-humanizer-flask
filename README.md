NLP Final Project — AI Content Forensics: Detect + Rewrite + Defend

This project is an end-to-end NLP system that combines classical NLP and the OpenAI API to:

detect AI-written vs human-written text,

detect formal vs informal tone,

rewrite text toward a target style (humanize / more informal), and

re-test the rewritten text to show whether the classification changed.

Educational project for an “Intelligent Language Systems / NLP” course.
The rewrite is transparent and grounded in the model’s own signals (top features + confidence).

Features (What the app does)

AI vs Human detector (TF-IDF + Logistic Regression)

Formality detector (TF-IDF + Logistic Regression)

Explainable signals: shows top contributing TF-IDF markers for the current input

OpenAI-powered rewrite loop:

Detect → Rewrite → Re-test

Shows before/after labels + confidence

Meaning preservation check using embeddings similarity

Neural baseline (BiGRU) for AI vs Human (report-only baseline)

Embedding visualization (2D t-SNE plot)

Models & Evaluation
AI vs Human — Classical (TF-IDF + Logistic Regression)

AI vs Human — Neural Baseline (BiGRU)

Formal vs Informal — Classical (TF-IDF + Logistic Regression)

Embedding Visualization (t-SNE, 2D)

Note: Very high scores can indicate the dataset contains strong stylistic cues. Generalization to other domains/detectors may vary (see Limitations).

OpenAI API Usage (3 different calls)

This project uses at least 3 OpenAI API calls:

Embeddings API — compute semantic similarity (meaning preservation) between original and rewritten text

LLM rewrite — rewrite text based on classifier outputs (labels, confidence, top markers)

LLM explanation — explain why the rewrite likely passed/failed based on before/after results

The LLM prompts receive the classical NLP outputs (top TF-IDF features + confidence), so rewriting is grounded in the model’s signals.

Tech Stack

Python

Flask (UI)

scikit-learn (TF-IDF, Logistic Regression, metrics)

TensorFlow/Keras (BiGRU neural baseline)

OpenAI API (rewrite + embeddings + explanation)

Project Structure
.
├── app.py
├── models/
│   ├── ai_vs_human_tfidf_logreg.joblib
│   └── formality_tfidf_logreg.joblib
├── templates/
│   └── index.html
├── assets/
│   ├── tf-idf-ai-vs-human.PNG
│   ├── neural-basline-ai-vs-human.PNG
│   ├── formal and informal graph.PNG
│   └── 2D-Graph.PNG
└── requirements.txt

How to Run (Local)
1) Create & activate a virtual environment

Windows (PowerShell)

python -m venv env
.\env\Scripts\Activate.ps1


Mac/Linux

python3 -m venv env
source env/bin/activate

2) Install dependencies
pip install -r requirements.txt

3) Set your OpenAI API key

Create a .env file in the project root:

OPENAI_API_KEY=your_key_here


Or set it as an environment variable.

4) Run Flask
python app.py


Open:

http://127.0.0.1:5000

Demo Plan (10 minutes)

Paste text → show AI/Human + Formality prediction + top markers

Click rewrite → show rewritten text

Show re-test results and confidence changes

Show embedding similarity score

Show evaluation plots + t-SNE figure

Discuss limitations + ethics

Datasets

AI vs Human Text (Kaggle) — used for AI/Human classifier + neural baseline

Formal & Informal Phrases (Kaggle) — used for formality classifier

TODO: Add Kaggle dataset names/links exactly as used in your notebook.

Ethical Considerations

Classifiers can produce false positives/negatives and may be biased toward the dataset’s writing styles.

User text should not be stored permanently; logs should avoid saving sensitive content.

This system is presented as an educational tool for NLP analysis and style editing.

Limitations & Next Steps

Detectors trained on one dataset may not generalize to other domains or third-party detectors (dataset shift).

Future improvements:

probability calibration

more diverse training data

stronger stylometry features (e.g., character n-grams, ensemble)

better UI for uncertainty (“uncertain” band)

Authors

Roman Mammadov