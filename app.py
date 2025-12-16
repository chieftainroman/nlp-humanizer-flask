import os
import joblib
import numpy as np
from flask import Flask, render_template, request
from dotenv import load_dotenv

from openai import OpenAI
load_dotenv()
app = Flask(__name__)

FORMALITY_PATH = "models/formality_tfidf_logreg.joblib"
AIHUMAN_PATH   = "models/ai_vs_human_tfidf_logreg.joblib"

formality_model = joblib.load(FORMALITY_PATH)
aihuman_model   = joblib.load(AIHUMAN_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def analyze_with_model(pipeline, text, top_k=12):
    tfidf = pipeline.named_steps["tfidf"]
    clf   = pipeline.named_steps["clf"]

    pred = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    classes = list(clf.classes_)
    conf = float(proba[classes.index(pred)])

 
    x = tfidf.transform([text])  

    weights = clf.coef_[0] 

    contrib = x.toarray()[0] * weights

    feature_names = np.array(tfidf.get_feature_names_out())
    present_idx = np.where(x.toarray()[0] > 0)[0]

    present_contrib = contrib[present_idx]
    order = np.argsort(np.abs(present_contrib))[::-1][:top_k]
    top_idx = present_idx[order]

    top_features = []
    for i in top_idx:
        top_features.append({
            "feature": feature_names[i],
            "contribution": float(contrib[i])
        })

    return {
        "label": pred,
        "confidence": conf,
        "top_features": top_features,
        "class_order": classes
    }


def cosine_sim(a, b):
    a = np.array(a); b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding


def gpt_rewrite(original_text, ai_stats, formality_stats, goal_human=True, goal_informal=True):
    ai_feats = [f["feature"] for f in ai_stats["top_features"]]
    form_feats = [f["feature"] for f in formality_stats["top_features"]]

    goals = []
    if goal_human:
        goals.append("Make the text more human-written (less LLM-like) while preserving meaning.")
    if goal_informal:
        goals.append("Make the tone slightly more informal and natural, but still polite.")
    goal_text = " ".join(goals) if goals else "Preserve meaning."

    system_msg = (
        "You are a writing assistant. "
        "You MUST preserve the original meaning and key facts. "
        "Rewrite only the style. Do not add new claims."
    )

    user_msg = f"""
ORIGINAL TEXT:
{original_text}

DETECTOR SIGNALS:
- AI/Human detector label: {ai_stats['label']} (confidence: {ai_stats['confidence']:.3f})
- AI/Human top markers in this text: {ai_feats}

- Formality detector label: {formality_stats['label']} (confidence: {formality_stats['confidence']:.3f})
- Formality top markers in this text: {form_feats}

TASK:
{goal_text}

CONSTRAINTS:
- Keep meaning the same.
- Avoid using the exact flagged markers too much.
- Output ONLY the rewritten text.
"""

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.7
    )
    return resp.output_text


def gpt_explain(original_text, rewritten_text, before_ai, after_ai, before_form, after_form, similarity):
    system_msg = (
        "You are an NLP teaching assistant. "
        "Explain results briefly and clearly. "
        "You must reference the detector labels/confidences and similarity score."
    )

    user_msg = f"""
ORIGINAL (short):
{original_text[:500]}

REWRITE (short):
{rewritten_text[:500]}

RESULTS:
- AI/Human BEFORE: {before_ai['label']} (conf {before_ai['confidence']:.3f})
- AI/Human AFTER:  {after_ai['label']} (conf {after_ai['confidence']:.3f})

- Formality BEFORE: {before_form['label']} (conf {before_form['confidence']:.3f})
- Formality AFTER:  {after_form['label']} (conf {after_form['confidence']:.3f})

- Embedding similarity (meaning preserved): {similarity:.3f}

Explain why it likely passed/failed and what stylistic changes mattered.
"""

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.2
    )
    return resp.output_text


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        goal_human = (request.form.get("goal_human") == "on")
        goal_informal = (request.form.get("goal_informal") == "on")

        if text:
            before_ai = analyze_with_model(aihuman_model, text)
            before_form = analyze_with_model(formality_model, text)

            rewritten = gpt_rewrite(
                text,
                ai_stats=before_ai,
                formality_stats=before_form,
                goal_human=goal_human,
                goal_informal=goal_informal
            )

            after_ai = analyze_with_model(aihuman_model, rewritten)
            after_form = analyze_with_model(formality_model, rewritten)

            emb1 = get_embedding(text)
            emb2 = get_embedding(rewritten)
            sim = cosine_sim(emb1, emb2)

            explanation = gpt_explain(text, rewritten, before_ai, after_ai, before_form, after_form, sim)

            result = {
                "original": text,
                "rewritten": rewritten,
                "before_ai": before_ai,
                "after_ai": after_ai,
                "before_form": before_form,
                "after_form": after_form,
                "similarity": sim,
                "explanation": explanation
            }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
