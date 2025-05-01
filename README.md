# sentiment-analysis-dashboard
Streamlit app for analyzing customer sentiment using Logistic Regression and TF-IDF.
# 📊 Sentiment Analysis Dashboard

An interactive Streamlit dashboard that analyzes customer product reviews and classifies them into Positive, Neutral, or Negative sentiments using Logistic Regression and TF-IDF.

---

## 🔍 Project Overview

This project was built as part of my data mining coursework. It allows users to explore sentiment trends, review patterns, and make real-time predictions through a clean and interactive interface.

---

## 🎯 Problem Statement

Businesses often struggle to manually analyze large volumes of customer feedback. This dashboard solves that by automating sentiment classification and visualizing insights clearly.

---

## 💡 Features

- 📈 Overview Tab: Total reviews, sentiment distribution, rating averages.
- ⏳ Timeline Tab: Sentiment trends over time (weekly/monthly/quarterly).
- 🔍 Explore Tab: Word clouds and price vs. rating analysis.
- 🤖 Prediction Tab: Enter any product review and get live sentiment prediction.

---

## 🛠 Tech Stack

- Python
- Streamlit
- scikit-learn
- Pandas, Matplotlib,Seaborn
- TF-IDF, Logistic Regression
- WordCloud, Joblib

---

## 📦 How to Run

```bash
pip install -r requirements.txt
streamlit run sentiment_analysis_dashboard.py
 sentiment-analysis-dashboard/

Project Structure
├── sentiment_analysis_dashboard.py
├── synthetic_product_reviews.csv
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md

🙋‍♂️ Author
Muhammad Aman
Third-Year Data Science Student
NED University of Engineering & Technology


