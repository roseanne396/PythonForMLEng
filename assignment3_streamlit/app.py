# assignment3_streamlit/app.py
# Streamlit UI for your HW2 FastAPI service

import io
import os
import json
import time
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Defaults (you can override in the sidebar or with env vars)
DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8004").rstrip("/")
DEFAULT_API_PATH = os.getenv("API_PATH", "/score_headlines")  # your POST route

st.set_page_config(page_title="Headline Scorer (Assignment 3)", layout="wide")


# ----------------------------- Helpers -----------------------------

def build_endpoint() -> str:
    base = st.session_state.get("api_base_url", DEFAULT_API_BASE).rstrip("/")
    path = st.session_state.get("api_path", DEFAULT_API_PATH)
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"


def call_api(headlines, endpoint):
    """
    POST headlines to the API and normalize the response to:
      [{"headline": ..., "label": ..., "score": ...}, ...]
    Supported server responses:
      - {"labels": [...]}
      - {"predictions": [{"headline":..., "label":..., "score":...}, ...]}
      - {"predictions": ["Optimistic", ...]}
      - {"labels":[...], "scores":[...]}
      - ["Optimistic", ...]
    """
    try:
        resp = requests.post(endpoint, json={"headlines": headlines}, timeout=45)
        resp.raise_for_status()
        data = resp.json()

        # Case: labels + scores side-by-side
        if isinstance(data, dict) and "labels" in data and "scores" in data:
            labels, scores = data["labels"], data["scores"]
            n = min(len(headlines), len(labels), len(scores))
            return [
                {"headline": headlines[i], "label": labels[i], "score": scores[i]}
                for i in range(n)
            ], None

        # Case: labels-only list
        if isinstance(data, dict) and isinstance(data.get("labels"), list):
            labels = data["labels"]
            n = min(len(headlines), len(labels))
            return [
                {"headline": headlines[i], "label": labels[i], "score": None}
                for i in range(n)
            ], None

        # Case: predictions already a list of dicts
        if isinstance(data, dict) and isinstance(data.get("predictions"), list) \
           and (not data["predictions"] or isinstance(data["predictions"][0], dict)):
            return data["predictions"], None

        # Case: predictions is a list of labels
        if isinstance(data, dict) and isinstance(data.get("predictions"), list) \
           and (not data["predictions"] or isinstance(data["predictions"][0], str)):
            labels = data["predictions"]
            n = min(len(headlines), len(labels))
            return [
                {"headline": headlines[i], "label": labels[i], "score": None}
                for i in range(n)
            ], None

        # Case: raw list of labels
        if isinstance(data, list) and (not data or isinstance(data[0], str)):
            labels = data
            n = min(len(headlines), len(labels))
            return [
                {"headline": headlines[i], "label": labels[i], "score": None}
                for i in range(n)
            ], None

        return None, f"Unexpected API response format: {data}"
    except requests.exceptions.RequestException as e:
        return None, f"Request error: {e}"


def parse_uploaded_file(uploaded_file):
    """Return a list of headline strings from txt/csv/json."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith(".txt"):
        text = raw.decode("utf-8", errors="ignore")
        return [ln.strip() for ln in text.splitlines() if ln.strip()]

    if name.endswith(".csv"):
        df = pd.read_csv(io.StringIO(raw.decode("utf-8", errors="ignore")))
        for col in ["headline", "headlines", "title", "text"]:
            if col in df.columns:
                return [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]
        # Fallback: first column
        return [str(x).strip() for x in df.iloc[:, 0].dropna().tolist() if str(x).strip()]

    if name.endswith(".json"):
        data = json.loads(raw.decode("utf-8", errors="ignore"))
        if isinstance(data, dict) and "headlines" in data and isinstance(data["headlines"], list):
            return [str(x).strip() for x in data["headlines"] if str(x).strip()]
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return [x.strip() for x in data if x.strip()]
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            out = []
            for item in data:
                for k in ("headline", "title", "text"):
                    if k in item and str(item[k]).strip():
                        out.append(str(item[k]).strip())
                        break
            return out
        raise ValueError("Unsupported JSON structure.")

    raise ValueError("Unsupported file type. Upload .txt, .csv, or .json.")


def ensure_session_state():
    if "headlines" not in st.session_state:
        st.session_state.headlines = [
            "Stocks rally as inflation cools",
            "Oil prices dip on supply concerns",
            "Tech shares extend gains",
        ]
    if "results_df" not in st.session_state:
        st.session_state.results_df = None
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = DEFAULT_API_BASE
    if "api_path" not in st.session_state:
        st.session_state.api_path = DEFAULT_API_PATH


ensure_session_state()


# ----------------------------- UI -----------------------------

st.title("Headline Sentiment Scorer")
st.caption("Edit or delete headlines in separate boxes, load from a file or pasted text, then send them to your Assignment 2 web service.")

with st.sidebar:
    st.header("Connection")
    st.session_state.api_base_url = st.text_input(
        "API base URL",
        value=st.session_state.api_base_url,
        help="Example: http://127.0.0.1:8004",
    )
    st.session_state.api_path = st.text_input(
        "API path",
        value=st.session_state.api_path,
        help="Example: /score_headlines or /predict",
    )
    st.write(f"Current endpoint: {build_endpoint()}")

    st.markdown("---")
    st.subheader("Load headlines from file")
    uploaded = st.file_uploader("Upload .txt, .csv, or .json", type=["txt", "csv", "json"])
    if uploaded is not None:
        try:
            loaded = parse_uploaded_file(uploaded)
            if loaded:
                st.session_state.headlines = loaded
                st.info(f"Loaded {len(loaded)} headlines from file.")
            else:
                st.warning("No headlines found in the file.")
        except Exception as e:
            st.error(f"Could not read file: {e}")

    st.subheader("Bulk paste (optional)")
    bulk = st.text_area("Paste one headline per line")
    if st.button("Split into boxes"):
        lines = [ln.strip() for ln in bulk.splitlines() if ln.strip()]
        if lines:
            st.session_state.headlines = lines
            st.info(f"Loaded {len(lines)} headlines from pasted text.")

st.subheader("Enter Headlines")

to_delete = []
for i, text in enumerate(st.session_state.headlines):
    c1, c2 = st.columns([8, 1])
    with c1:
        st.session_state.headlines[i] = st.text_input(
            f"Headline {i+1}",
            value=text,
            key=f"headline_{i}",
        )
    with c2:
        if st.button("Delete", key=f"delete_{i}", help="Delete this headline"):
            to_delete.append(i)

# Remove selected
for idx in sorted(to_delete, reverse=True):
    st.session_state.headlines.pop(idx)

if st.button("Add headline"):
    st.session_state.headlines.append("")

st.markdown("---")
left, right = st.columns([1, 1])
with left:
    do_score = st.button("Score Headlines")
with right:
    if st.button("Clear results"):
        st.session_state.results_df = None
        st.info("Results cleared.")


# ----------------------------- Scoring -----------------------------

if do_score:
    headlines = [h.strip() for h in st.session_state.headlines if h.strip()]
    if not headlines:
        st.warning("Please enter at least one non-empty headline.")
    else:
        endpoint = build_endpoint()
        with st.spinner("Scoring..."):
            results, err = call_api(headlines, endpoint)
            if err:
                st.error(err)
            else:
                df = pd.DataFrame(results)
                # Normalize expected columns
                for col in ["headline", "label", "score"]:
                    if col not in df.columns:
                        df[col] = None
                st.session_state.results_df = df[["headline", "label", "score"]]
                st.success(f"Received {len(df)} predictions.")


# ----------------------------- Results -----------------------------

if st.session_state.results_df is not None:
    st.subheader("Results")
    st.dataframe(st.session_state.results_df, use_container_width=True)
    csv = st.session_state.results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"headline_scores_{int(time.time())}.csv",
        mime="text/csv",
    )
