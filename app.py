import re
from pathlib import Path

import joblib
import nltk
import numpy as np
import streamlit as st
from nltk.stem import WordNetLemmatizer

MODEL_PATH = Path(__file__).parent / "fake_news_model.joblib"

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


@st.cache_resource
def load_artifacts():
    """Load the trained models + the exact stopword set used in training."""
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    artifacts = joblib.load(MODEL_PATH)
    return artifacts, WordNetLemmatizer()


def preprocess(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    """Same pipeline as the notebook: lowercase -> letters only -> stopwords -> lemmatize."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(lemmatizer.lemmatize(w) for w in words)


def predict(clean_text: str, artifacts: dict, model_key: str):
    vec = artifacts["vectorizer"].transform([clean_text])
    model = artifacts[model_key]
    pred = int(model.predict(vec)[0])  # 1 = real, 0 = fake

    if model_key == "lr":
        p_real = float(model.predict_proba(vec)[0][1])
    else:  # LinearSVC has no probabilities, squash the margin into 0-1
        p_real = float(1 / (1 + np.exp(-model.decision_function(vec)[0])))
    return pred, p_real, vec


def top_terms(vec, model, feature_names, k=8):
    """Words in this article that pushed the model most toward real / fake."""
    contrib = vec.data * model.coef_[0][vec.indices]
    order = np.argsort(contrib)
    words = feature_names[vec.indices]
    fake = [words[i] for i in order[:k] if contrib[i] < 0]
    real = [words[i] for i in order[::-1][:k] if contrib[i] > 0]
    return real, fake


# ----------------------------- UI -----------------------------
st.title("📰 Fake News Detector")
st.caption("TF-IDF + Logistic Regression / Linear SVM trained on the Fake and Real News dataset.")

if not MODEL_PATH.exists():
    st.error(
        f"`{MODEL_PATH.name}` not found. Save it from the notebook and put it next to app.py."
    )
    st.stop()

artifacts, lemmatizer = load_artifacts()

with st.sidebar:
    st.header("Settings")
    model_choice = st.radio(
        "Model",
        options=["Logistic Regression", "Linear SVM"],
        help="Logistic Regression gives real probabilities. SVM is slightly more accurate on the test set.",
    )
    model_key = "lr" if model_choice == "Logistic Regression" else "svm"

title = st.text_input("Headline (optional)")
body = st.text_area("Article text", height=250, placeholder="Paste the news article here...")

if st.button("Analyze", type="primary", use_container_width=True):
    full_text = f"{title} {body}".strip()

    if len(full_text.split()) < 30:
        st.warning("Please enter at least ~30 words. Short text gives unreliable results.")
        st.stop()

    clean = preprocess(full_text, artifacts["stop_words"], lemmatizer)
    if not clean:
        st.warning("No usable words left after cleaning. Try a different text.")
        st.stop()

    pred, p_real, vec = predict(clean, artifacts, model_key)
    confidence = p_real if pred == 1 else 1 - p_real

    if pred == 1:
        st.success("✅ Looks like **REAL** news")
    else:
        st.error("🚨 Looks like **FAKE** news")

    col1, col2 = st.columns(2)
    col1.metric("Confidence", f"{confidence:.1%}")
    col2.metric("Real-news score", f"{p_real:.1%}")
    st.progress(p_real)
    if model_key == "svm":
        st.caption("SVM score is not a calibrated probability. Use it as a rough indication.")

    real_words, fake_words = top_terms(
        vec, artifacts[model_key], artifacts["vectorizer"].get_feature_names_out()
    )
    with st.expander("Why? Words that influenced the decision"):
        c1, c2 = st.columns(2)
        c1.markdown("**Pushing toward REAL**")
        c1.write(", ".join(real_words) or "-")
        c2.markdown("**Pushing toward FAKE**")
        c2.write(", ".join(fake_words) or "-")

with st.expander("⚠️ Limitations"):
    st.markdown(
        """
- The model was trained on US political news from 2016-2017: real articles come from one news agency
  and fake ones from a few websites.
- It learns **writing style and topics** of those sources, not whether the claims are actually true.
- It may perform much worse on other topics, other outlets, or newer events.
- Do not use it as a fact-checker. Verify important claims with trusted sources.
"""
    )
