"""Small shared UI/client helpers; endpoint addresses are operator-controlled only."""
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import httpx
import streamlit as st


def setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    try:
        return str(st.secrets.get(name, default))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return default


def page(title: str, subtitle: str):
    st.set_page_config(page_title=title + " · AI Portfolio Projects", page_icon="◈", layout="wide")
    st.caption("AI PORTFOLIO PROJECTS / DEPLOYABLE REFERENCE APPS")
    st.title(title)
    st.write(subtitle)
    st.sidebar.caption("No model results are fabricated. Remote inference requires a configured backend.")


def api_url() -> str:
    url = setting("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("API_BASE_URL must use HTTP(S)")
    return url


def headers() -> dict:
    key = setting("API_KEY")
    return {"Authorization": "Bearer " + key} if key else {}


def api_get(path: str):
    response = httpx.get(api_url() + path, headers=headers(), timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(path: str, body: dict):
    response = httpx.post(api_url() + path, headers=headers(), json=body, timeout=30)
    response.raise_for_status()
    return response.json()


REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_json(relative: str) -> dict:
    """Read a committed repository artifact (for example a checked-in evaluation report)."""
    path = (REPO_ROOT / relative).resolve()
    if REPO_ROOT not in path.parents or not path.is_file():
        raise ValueError(f"Committed artifact missing: {relative}")
    return json.loads(path.read_text())


def uploaded_json(file):
    if file is None:
        return None
    if file.size > 5 * 1024 * 1024:
        raise ValueError("Maximum report size is 5 MB")
    data = json.loads(file.getvalue())
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


def unavailable():
    st.error("Backend unavailable, unauthorized, or returned an invalid response. Check server configuration.")
    st.info("Set API_BASE_URL and API_KEY in the deployment secrets. On Streamlit Cloud, localhost is the "
            "cloud machine—not your laptop. See this project's deployment guide.")
