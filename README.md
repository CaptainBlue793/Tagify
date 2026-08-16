# Tagify ☴

**Paste in any block of text — get back its hidden topics and the industries it belongs to.** Unsupervised topic
modelling plus industry classification, in one Streamlit page.

🚀 **Live demo:** [huggingface.co/spaces/Scarletta975/Tagify](https://huggingface.co/spaces/Scarletta975/Tagify)

---

## What it does

Tagify runs two independent passes over your text and shows both results side by side:

**1. Topic modelling (unsupervised)**
Tokenizes and strips stop words with `gensim`, builds a dictionary and bag-of-words corpus, then trains an
**LDA** (Latent Dirichlet Allocation) model on the fly. It surfaces 5 topics, each described by its 10 most
probable words. Nothing is pre-trained — the model is fitted to whatever you paste, so the topics are specific to
that text.

**2. Industry labelling (keyword-based)**
Scores the text against a curated taxonomy of **30 industries** — Finance, Healthcare, Law, Telecom, Gaming,
Aviation, Agriculture and more — each backed by a hand-built keyword list. Every whole-word keyword match scores a
point, and the top 5 industries are returned.

The two are deliberately complementary: LDA tells you *what this text is about in its own words*, the taxonomy
tells you *which industry bucket it lands in*.

## Repo layout

| File | What it is |
|------|-----------|
| `Tagify.py` | Streamlit app — preprocessing, LDA training, industry scoring, UI |
| `Tags.py` | The industry taxonomy: 30 industries and their keyword lists |
| `Images/` | Background art |
| `config.toml` | Streamlit dark theme |
| `font.css` | Custom font styling |
| `PythonPackages.txt` | Dependency list |

## Running it locally

```bash
git clone https://github.com/CaptainBlue793/Tagify.git
cd Tagify

pip install streamlit gensim pandas

streamlit run Tagify.py
```

Paste text into the box, hit **Analyze Text**, and the topics and industry labels appear on the right.

## Extending it

Adding an industry is a two-line change in `Tags.py` — define a keyword list, then register it in the `industries`
dictionary:

```python
robotics_keywords = ["robotics", "actuator", "manipulator", "SLAM", "ROS", "end effector"]

industries = {
    ...
    "Robotics": robotics_keywords,
}
```

To change how many topics or words per topic are produced, adjust `num_topics` / `num_words` in
`perform_topic_modeling()`.

## Notes & limitations

- LDA on a **single short document** is noisy — it works far better on a long transcript or article than on a
  couple of sentences.
- Industry labelling is exact keyword matching, so it has no notion of synonyms or context.
- The top-5 list is returned even when scores are zero, so low-signal text can still produce arbitrary-looking labels.

## Tech stack

Python · Gensim (LDA) · Streamlit
