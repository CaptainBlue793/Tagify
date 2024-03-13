"""Topic Modelling and Labelling App"""

import base64
import heapq
import re

# Importing packages
import gensim
import streamlit as st
from gensim import corpora, models
from Tags import industries


# ...
# Topic Modelling on a given text
def preprocess_text(text):
    # Replace this with your own preprocessing code
    # This example simply tokenizes the text and removes stop words
    tokens = gensim.utils.simple_preprocess(text)
    stop_words = gensim.parsing.preprocessing.STOPWORDS
    preprocessed_text = [[token for token in tokens if token not in stop_words]]
    return preprocessed_text


def perform_topic_modeling(transcript_text, num_topics=5, num_words=10):
    # Preprocess the transcript text
    # Replace this with your own preprocessing code
    preprocessed_text = preprocess_text(transcript_text)
    # Create a dictionary of all unique words in the transcripts
    dictionary = corpora.Dictionary(preprocessed_text)
    # Convert the preprocessed transcripts into a bag-of-words representation
    corpus = [dictionary.doc2bow(text) for text in preprocessed_text]
    # Train an LDA model with the specified number of topics
    lda_model = models.LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=num_topics
    )
    # Extract the most probable words for each topic
    Topics = []
    for idx, Topic in lda_model.print_topics(-1, num_words=num_words):
        # Extract the top words for each topic and store in a list
        topic_words = [
            word.split("*")[1].replace('"', "").strip() for word in Topic.split("+")
        ]
        Topics.append((f"Topic {idx}", topic_words))
    return Topics


def label_topic(labelling_text):
    """
    Given a piece of text, this function returns the top five industry labels that best match the topics discussed
    in the text.
    """
    # Count the number of occurrences of each keyword in the text for each industry
    counts = {}
    for industry, keywords in industries.items():
        count = sum(
            [
                1
                for keyword in keywords
                if re.search(r"\b{}\b".format(keyword), labelling_text, re.IGNORECASE)
            ]
        )
        counts[industry] = count
    # Get the top five industries based on their counts
    top_industries = heapq.nlargest(5, counts, key=counts.get)

    # If only one industry was found, return it
    if len(top_industries) == 1:
        return top_industries[0]
    # If five industries were found, return them both
    else:
        return top_industries


# ...

# Streamlit Code
st.set_page_config(layout="wide")

# Font Style
with open("font.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


# Display Background
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


add_bg_from_local("Images/background.png")
# Main content
st.markdown(
    """
    <style>
    .tagify-title {
        font-size: 62px;
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    .tagify-title span {
        transition: color 0.2s ease-in-out;
    }
    .tagify-title:hover span {
        color: #f5fefd; /* Hover color */
    }
    .tagify-title:hover {
        transform: scale(1.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

text = "Tagify"  # Text to be styled
colored_text = ''.join(
    ['<span style="color: hsl({}, 70%, 50%);">{}</span>'.format(20 + (i * 30 / len(text)), char) for i, char in
     enumerate(text)])
colored_text_with_malt = colored_text + ' <span style="color: hsl(40, 70%, 50%);">&#9778;</span>'
st.markdown(f'<h1 class="tagify-title">{colored_text_with_malt}</h1>', unsafe_allow_html=True)

st.markdown(
    '<h2 style="font-size:30px;color: #F5FEFD; text-align: center;">Topic Modelling and Labelling</h2>',
    unsafe_allow_html=True,
)

input_text = st.text_area("Paste your Input Text", height=200)
if st.button("Analyze Text"):
    col1, col2 = st.columns([2, 2])
    with col1:
        st.info("Text is below")
        st.write(input_text)
    with col2:
        # Perform topic modeling on the transcript text
        topics = perform_topic_modeling(input_text)
        # Display the resulting topics in the app
        st.info("Topics in the Text")
        for topic in topics:
            st.success(f"{topic[0]}: {', '.join(topic[1])}", icon="✅")
        # Label the text with the top five industries
        label = label_topic(input_text)
        st.info("Top Five Industries")
        st.success(f"{', '.join(label)}", icon="✅")

