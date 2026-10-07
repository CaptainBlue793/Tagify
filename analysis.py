"""Deterministic topic discovery and transparent industry matching."""
from collections import Counter
import re

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from Tags import industries

MAX_CHARACTERS = 300_000


def tokenize(text):
    return [word for word in re.findall(r"\b[a-zA-Z][a-zA-Z'-]{2,}\b", text.lower()) if word not in ENGLISH_STOP_WORDS]


def parse_taxonomy(text):
    custom = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        name, separator, values = line.partition(":")
        keywords = [value.strip() for value in values.split(",") if value.strip()]
        if not separator or not name.strip() or not keywords:
            raise ValueError("Use Industry: keyword, another keyword for each custom industry.")
        custom[name.strip()] = keywords
    return custom


def score_industries(text, custom=None):
    results = []
    for name, keywords in {**industries, **(custom or {})}.items():
        matches = {}
        for keyword in sorted(set(word.lower().strip() for word in keywords if word.strip())):
            phrase = r"\s+".join(re.escape(part) for part in keyword.split())
            count = len(re.findall(r"(?<!\w)" + phrase + r"(?!\w)", text, flags=re.IGNORECASE))
            if count:
                matches[keyword] = count
        if matches:
            results.append({"industry": name, "score": len(matches), "occurrences": sum(matches.values()), "matched_keywords": list(matches)})
    return sorted(results, key=lambda row: (-row["score"], -row["occurrences"], row["industry"]))


def analyze(text, num_topics=4, num_words=6, custom=None):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Add some text before analyzing.")
    if len(text) > MAX_CHARACTERS:
        raise ValueError("Use at most 300,000 characters per analysis.")
    if not 1 <= num_topics <= 8 or not 3 <= num_words <= 12:
        raise ValueError("Choose 1–8 topics and 3–12 words per topic.")
    tokens = tokenize(text)
    if not tokens:
        raise ValueError("Add text with meaningful English words; punctuation or stop words alone cannot form topics.")
    counts = Counter(tokens)
    segments = []
    for paragraph in re.split(r"\n\s*\n|(?<=[.!?])\s+", text):
        words = tokenize(paragraph)
        segments.extend(" ".join(words[i:i + 100]) for i in range(0, len(words), 100) if words[i:i + 100])
    mode = "keywords"
    topics = [{"name": "Key themes", "words": [word for word, _ in counts.most_common(num_words)], "weight": 1.0}]
    if len(tokens) >= 60 and len(segments) >= 3 and len(counts) >= 12:
        vectorizer = CountVectorizer(max_features=4000, stop_words="english")
        matrix = vectorizer.fit_transform(segments)
        n_topics = min(num_topics, len(segments), max(1, matrix.shape[1] // 5))
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=15, learning_method="batch")
        weights = model.fit_transform(matrix).mean(axis=0)
        terms = vectorizer.get_feature_names_out()
        topics = [{"name": f"Topic {rank + 1}", "words": terms[np.argsort(model.components_[index])[::-1][:num_words]].tolist(), "weight": float(weights[index])} for rank, index in enumerate(np.argsort(weights)[::-1])]
        mode = "lda"
    return {"source": text.strip(), "word_count": len(text.split()), "unique_terms": len(counts), "segments": len(segments), "mode": mode, "topics": topics, "keywords": [{"term": term, "count": count} for term, count in counts.most_common(15)], "industries": score_industries(text, custom)}
