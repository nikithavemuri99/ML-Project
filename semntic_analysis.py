import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download("vader_lexicon")
nltk.download("stopwords")

# Load dataset
data_path = "sample data.csv"
df = pd.read_csv(data_path)

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\@w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

df["cleaned_text"] = df["text"].apply(clean_text)

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
df["sentiment"] = df["cleaned_text"].apply(lambda x: sia.polarity_scores(x)["compound"])

# Label Sentiment
df["sentiment_label"] = df["sentiment"].apply(lambda x: "Positive" if x > 0.05 else ("Negative" if x < -0.05 else "Neutral"))

# Word Cloud
def plot_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

plot_wordcloud(df["cleaned_text"])

# Save processed data
df.to_csv("processed_data.csv", index=False)

print("Semantic Analysis Completed. Results saved to processed_data.csv")
