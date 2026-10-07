# Tagify

A text intelligence workspace for discovering themes and explaining industry labels. Paste an article or transcript, upload UTF-8 text, or try the built-in example. No model downloads, API keys, or GPU required.

## What's inside

- Deterministic LDA over non-overlapping text segments, rather than one bag of words for the entire input.
- Key-term fallback for short or low-signal text; no invented multi-topic result.
- Thirty industry dictionaries with escaped, case-insensitive, whole-phrase matching. Zero-score labels are omitted.
- Matched keywords, occurrence counts, prominent-term charts, and editable topic settings.
- Additional industry dictionaries through `Industry: keyword, another keyword` in the sidebar. A matching industry name replaces its default dictionary for that analysis.
- Session-persistent results, source inspection, and JSON/CSV exports.

## Run locally with starGPU

From this directory in PowerShell:

```powershell
conda activate starGPU
python -m pip install -r requirements.txt
python -m streamlit run Tagify.py --server.port 8501
```

Alternatively, `./run.ps1` launches through `conda run -n starGPU` without needing to activate the environment first. Open http://localhost:8501. The standard Streamlit theme lives in `.streamlit/config.toml`.

## Analysis behavior

Topic analysis is English-oriented and uses scikit-learn stop words. Inputs with at least 60 meaningful tokens, 3 segments, and 12 distinct terms use LDA; smaller inputs show frequent key terms. The requested topic count is an upper bound, constrained by available evidence. A fixed random seed makes identical inputs reproducible. Topic weights describe the fitted model, not factual confidence.

Industry scores count distinct matching keywords, with occurrence counts shown separately. Related industries can overlap. Matching is lexical: it does not infer synonyms, resolve ambiguity, or establish that a document belongs to an industry. There are no arbitrary zero-score top-five labels.

Text is limited to 300,000 characters; uploads to 2 MB. Uploaded text takes precedence over pasted text and is not saved to disk. Results represent the last submitted analysis until you analyze again.

## Files and checks

`Tagify.py` contains the interface and compatibility helpers; `analysis.py` contains the independently testable engine; `Tags.py` contains the existing taxonomy; `style.css` contains visual styling. Legacy artwork, `font.css`, and the root `config.toml` remain available but are not used by the new interface. `PythonPackages.txt` redirects to the actual installable requirements.

```powershell
python -m pip install pytest
python -m pytest tests -q
```

Tests cover validation, escaped phrases, unmatched inputs, deterministic topic discovery, and Streamlit interaction/persistence.
