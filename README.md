Hate Speech Detection and Classification
Overview
This project focuses on detecting and classifying hate speech in tweets. The goal is to preprocess and analyze text data to identify whether a tweet contains hate speech. The project utilizes text processing techniques, sentiment analysis, and machine learning for classification.

Project Structure
Data: HateSpeech_Kenya.csv - Dataset containing tweets labeled with different classes including hate speech, offensive language, and neutral.
Script: A Python script that includes data preprocessing, sentiment analysis, and text classification.
Dependencies
Ensure you have the following Python packages installed:

pandas
nltk
textblob
seaborn
matplotlib
scikit-learn
vaderSentiment
transformers
datasets
torch
You can install these dependencies using:


bash
Copy code
pip install pandas nltk textblob seaborn matplotlib scikit-learn vaderSentiment transformers datasets torch
Setup
Download NLTK stopwords:

python
Copy code
import nltk
nltk.download('stopwords')
Load the dataset:

Ensure that the dataset HateSpeech_Kenya.csv is located in the /content/ directory.

Preprocess the Data:

The preprocessing involves cleaning the text, removing URLs, mentions, hashtags, and non-alphanumeric characters. It also converts the text to lowercase and removes stop words.

Sentiment Analysis:

Sentiment analysis is performed using the TextBlob library, calculating polarity and subjectivity of each tweet.

Classification:

Based on the sentiment polarity, the text is classified as either "Hate Speech" or "Non-Hate Speech".

Code
Data Loading and Preprocessing
python
Copy code
import pandas as pd
import re
from nltk.corpus import stopwords
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk

# Download stopwords
nltk.download('stopwords')

# Load dataset
logue1 = '/content/HateSpeech_Kenya.csv'
dlogue = pd.read_csv(logue1)

# Preprocess text data
stop_words = set(stopwords.words('english'))

def janitor(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def hulk(texty):
    blob = TextBlob(texty)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Process tweets
line_up = list(dlogue['Tweet'])
Preprocessed_tweets = [janitor(me) for me in line_up]
dlogue['Processed_tweets'] = Preprocessed_tweets
Sentiment Analysis and Classification
python
Copy code
dlogue[['polarity', 'subjectivity']] = dlogue['Processed_tweets'].apply(hulk).apply(pd.Series)
dlogue['hate_speech'] = dlogue['polarity'].apply(lambda x: 1 if x < 0 else 0)

# Classification metrics
accuracy = accuracy_score(dlogue['neither'], dlogue['hate_speech'])
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(dlogue['hate_speech'], dlogue['hate_speech'], target_names=['Non-Hate Speech', 'Hate Speech'])
print("Classification Report:\n", report)

confmatrix = confusion_matrix(dlogue['neither'], dlogue['hate_speech'])
plt.figure(figsize=(8, 6))
sns.heatmap(confmatrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Non-Hate Speech', 'Predicted Hate Speech'],
            yticklabels=['Actual Non-Hate Speech', 'Actual Hate Speech'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
Text Classification Function
python
Copy code
def classify_text(text):
    processed_text = janitor(text)
    blob = TextBlob(processed_text)
    polarity = blob.sentiment.polarity
    if polarity < 0:
        return "Hate Speech"
    else:
        return "Non-Hate Speech"

# Example classification
text_to_classify = "She ate apples"
classification = classify_text(text_to_classify)
print(f"The text is classified as: {classification}")
Results
Accuracy: The model achieved an accuracy of approximately 11%. This indicates that the model's classification may not be very reliable, and further tuning and model improvements may be required.
Classification Report: Provides precision, recall, and F1-score metrics for both classes ("Hate Speech" and "Non-Hate Speech").
Confusion Matrix: Visual representation of the classification results.
License
This project is licensed under the MIT License - see the LICENSE file for details.

