# Fake News Detection

An NLP-based Machine Learning application that analyzes news articles and predicts whether they are **Real** or **Fake** using TF-IDF and trained classification models.

The project includes text preprocessing, feature extraction, model training, evaluation, and an interactive web application built with Streamlit.

## Project Overview

Fake news can spread quickly through digital platforms. This project explores how Natural Language Processing and Machine Learning can help identify patterns in news articles and classify them as real or fake.

**Important:** The model identifies patterns learned from its training dataset. It does not verify whether a news claim is factually true.

## Features

- Text preprocessing using NLTK.
- TF-IDF feature extraction.
- Logistic Regression and Linear SVM models.
- Interactive Streamlit interface.
- Headline and article-text input.
- Prediction confidence and influential words.
- Model loading using Joblib.

## Dataset

The project uses the Fake and Real News Dataset, containing two CSV files:

- `Fake.csv` — fake news articles.
- `True.csv` — real news articles.

The dataset was combined, labeled, and cleaned before model training.

**Label mapping:**
- `0` = Real News
- `1` = Fake News

The dataset files are not included in this repository.

## NLP Pipeline

1. Combine the article headline and body.
2. Convert text to lowercase.
3. Remove punctuation, numbers, and non-English characters.
4. Remove stopwords while preserving negation words where applicable.
5. Apply lemmatization using NLTK.
6. Convert text into numerical features using TF-IDF.

## Machine Learning Models

The application is designed to support:

- Logistic Regression
- Linear Support Vector Machine (Linear SVM)

Both models use TF-IDF features generated from the preprocessed text.

## Repository Structure

```text
Fake-News-Detection/
│
├── app.py
├── fake-news-detection-nltk-logistic-reg-svc__.ipynb
├── fake_news_model.joblib
├── requirements.txt
└── README.md
