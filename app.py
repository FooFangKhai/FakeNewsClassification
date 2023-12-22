import streamlit as st
import pickle
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from summarizer import Summarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

st.set_page_config(
    page_title="Fake News Classification App",
    page_icon=":newspaper:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("<h1 style='text-align: center; color: #0071EB;'>Fake News Classification, Summarization, and Keywords Extraction App</h1>", unsafe_allow_html=True)
st.markdown("---")
st.write("Created by Foo Fang Khai")

lm = WordNetLemmatizer()
vectorization = TfidfVectorizer()

vector_form = pickle.load(open("vector.pkl", "rb"))
load_model = pickle.load(open("rfc.pkl", "rb"))

def lemmatization(content):
    con = re.sub(r'[^a-zA-Z0-9]', ' ', content)
    con = con.lower()
    words = word_tokenize(con)
    con = [lm.lemmatize(word) for word in words if word not in stopwords.words('english')]
    con = " ".join(con)
    return con

def fake_news(news):
    news = lemmatization(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction

def summarize_text(text):
    summarizer = Summarizer()
    summary = summarizer(text, ratio=0.2)
    return summary

def extract_keywords(text):
    words = word_tokenize(text)
    words = [lm.lemmatize(word) for word in words if word not in stopwords.words('english')]
    preprocessed_text = " ".join(words)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_text])
    feature_names = tfidf_vectorizer.get_feature_names_out()

    top_keywords_indices = tfidf_matrix.toarray()[0].argsort()[-15:][::-1]
    top_keywords = [feature_names[i] for i in top_keywords_indices]

    return top_keywords

sentence = st.text_area("Enter your news content here", "", height=200, key='text_area')
st.markdown('<style>div.Widget.row-widget.stTextArea {background-color: #f0f0f0;}</style>', unsafe_allow_html=True)

predict_btn = st.button("Predict", key='predict_btn')
st.markdown(
    f'<style>div.stButton > button {{background-color: #0071EB; color: white; width: 100%; padding: 0.5rem; border-radius: 5px; cursor: pointer;}}</style>',
    unsafe_allow_html=True)
summarize_button = st.button("Summarize", key='summarize_button')
st.markdown(
    f'<style>div.stButton > button {{background-color: #00C96F; color: white; width: 100%; padding: 0.5rem; border-radius: 5px; cursor: pointer;}}</style>',
    unsafe_allow_html=True)
extract_keywords_button = st.button("Extract Keywords", key='extract_keywords_button')
st.markdown(
    f'<style>div.stButton > button {{background-color: #FFA500; color: white; width: 100%; padding: 0.5rem; border-radius: 5px; cursor: pointer;}}</style>',
    unsafe_allow_html=True)

if predict_btn:
    prediction_class = fake_news(sentence)
    if prediction_class == [0]:
        st.success('Reliable')
    if prediction_class == [1]:
        st.error('Unreliable')

if summarize_button:
    summary = summarize_text(sentence)
    st.subheader("Summarized News")
    st.write(summary)

if extract_keywords_button:
    top_keywords = extract_keywords(sentence)
    st.subheader("Top Keywords Extracted (based on TF-IDF)")
    st.write(top_keywords)

if "top_keywords" in locals():
    lemmatized_text = lemmatization(sentence)
    st.subheader("Lemmatized Text")
    st.write(lemmatized_text)

if sentence:
    original_word_count = len(sentence.split())

    if "summary" in locals() and "top_keywords" not in locals():
        summarized_word_count = len(summary.split())
        st.write(f"Original Word Count: {original_word_count}")
        st.write(f"Summarized Word Count: {summarized_word_count}")