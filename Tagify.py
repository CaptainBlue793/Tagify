"""Tagify's Streamlit workspace. Run with streamlit run Tagify.py."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis import analyze, parse_taxonomy

SAMPLE = """Hospitals are adopting telemedicine and electronic health records to improve patient care.
Doctors and nurses use medical devices to monitor patients remotely. Clinical trials evaluate new medicine.

Machine learning and artificial intelligence help software teams analyze healthcare data.
Cloud computing supports secure data analytics, while cybersecurity protects patient information.

Renewable energy and solar energy can reduce a hospital's carbon footprint.
Sustainability and energy efficiency are becoming part of healthcare policy.

Software engineering teams build secure databases and deploy cloud services for clinical research.
Pharmaceutical researchers study drug development and personalized medicine through biotechnology.
Solar panels and smart grid systems help hospitals manage electricity consumption and renewable resources."""


def preprocess_text(text):
    from analysis import tokenize
    return [tokenize(text)]


def perform_topic_modeling(transcript_text, num_topics=5, num_words=10):
    return [(t["name"], t["words"]) for t in analyze(transcript_text, num_topics, num_words)["topics"]]


def label_topic(text):
    from analysis import score_industries
    return [row["industry"] for row in score_industries(text)[:5]]


def main():
    st.set_page_config(page_title="Tagify · Text intelligence", page_icon="🏷️", layout="wide")
    st.markdown(f"<style>{Path(__file__).with_name('style.css').read_text()}</style>", unsafe_allow_html=True)
    st.caption("TAGIFY / TEXT INTELLIGENCE")
    st.title("Find the signal in your text.")
    st.write("Explore themes, discover industries, and inspect the words behind every match.")
    with st.sidebar:
        st.header("Analysis settings")
        topic_count = st.slider("Maximum topics", 1, 8, 4)
        word_count = st.slider("Words per topic", 3, 12, 6)
        limit = st.slider("Industry results", 1, 15, 5)
        custom = st.text_area("Additional industries", placeholder="Robotics: robot, actuator, robotics", help="One industry per line; separate its keywords with commas.")
        st.caption("English topic analysis. Industry matches are keyword evidence, not probabilities.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Your source")
        if st.button("Try an example"):
            st.session_state["source"] = SAMPLE
        upload = st.file_uploader("Or upload text", type=["txt", "md"])
        source = st.text_area("Text to analyze", key="source", height=300, max_chars=300_000)
        if upload is not None:
            if upload.size > 2_000_000:
                st.error("Choose a text file smaller than 2 MB.")
                source = ""
            else:
                try:
                    source = upload.getvalue().decode("utf-8-sig")
                    st.caption(f"Using uploaded file: {upload.name}")
                except UnicodeDecodeError:
                    st.error("Save your text file as UTF-8 and upload it again.")
                    source = ""
        if st.button("Analyze text", type="primary", use_container_width=True):
            try:
                with st.spinner("Discovering themes and matching industries…"):
                    st.session_state["analysis"] = analyze(source, topic_count, word_count, parse_taxonomy(custom))
            except ValueError as exc:
                st.error(str(exc))
        st.caption("Uploaded text is not written to disk.")
    with right:
        st.subheader("A clearer view")
        st.info("Start with an article, transcript, or research note. Longer text gives topic modeling more evidence.")
        st.markdown("**01 · Themes**  \nRepeatable topic analysis with meaningful keywords.\n\n**02 · Industry evidence**  \nSee exactly which phrases matched.\n\n**03 · Take it with you**  \nDownload structured results for your next workflow.")
    result = st.session_state.get("analysis")
    if not result:
        return
    st.divider()
    metrics = st.columns(4)
    for col, label, value in zip(metrics, ["Words", "Distinct terms", "Text segments", "Industry matches"], [result["word_count"], result["unique_terms"], result["segments"], len(result["industries"])]):
        col.metric(label, value)
    st.caption("Results reflect the last analysis. Analyze again after editing the source or settings.")
    themes, labels, details = st.tabs(["Themes", "Industries", "Export & source"])
    with themes:
        if result["mode"] == "keywords":
            st.info("This source has limited topic evidence. Showing key terms instead of fitting several topics.")
        for topic in result["topics"]:
            with st.container(border=True):
                st.subheader(topic["name"])
                st.write(" · ".join(topic["words"]))
                st.caption(f"Share of modeled topic weight: {topic['weight']:.0%}")
        st.subheader("Prominent terms")
        st.bar_chart(pd.DataFrame(result["keywords"]).set_index("term")["count"], color="#f4b860", horizontal=True)
    with labels:
        rows = result["industries"][:limit]
        if not rows:
            st.info("No industry keywords matched. Add a relevant industry in the sidebar or try a longer source.")
        for row in rows:
            with st.container(border=True):
                st.markdown(f"**{row['industry']}** — {row['score']} distinct keyword matches")
                st.write(", ".join(row["matched_keywords"]))
                st.caption(f"Total occurrences: {row['occurrences']}")
    with details:
        a, b = st.columns(2)
        a.download_button("Download analysis JSON", json.dumps(result, indent=2, ensure_ascii=False), "tagify-analysis.json", "application/json")
        b.download_button("Download industry CSV", pd.DataFrame(result["industries"], columns=["industry", "score", "occurrences", "matched_keywords"]).to_csv(index=False), "tagify-industries.csv", "text/csv")
        st.text_area("Analyzed source", result["source"], height=220, disabled=True)


if __name__ == "__main__":
    main()
