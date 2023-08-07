import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load data from CSV file into pandas DataFrame
df = pd.read_csv('Comments_Data.csv')

# Define function to preprocess text
def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    text = "".join([char.lower() for char in text if char.isalpha() or char==" "])
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Apply text preprocessing to comment text column
df['processed_comment'] = df['comment_body'].apply(lambda x: preprocess_text(x))
print(df)

# Define function to perform sentiment analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply sentiment analysis to processed comment text column
df['sentiment'] = df['processed_comment'].apply(lambda x: get_sentiment(x))

# Define function to perform emotion detection
def get_emotion(text):
    blob = TextBlob(text)
    emotions = {"anger": 0, "joy": 0, "sadness": 0, "fear": 0}
    for sentence in blob.sentences:
        emotion = sentence.sentiment.polarity
        if emotion >= 0.5:
            emotions["joy"] += 1
        elif emotion > 0:
            emotions["joy"] += emotion
        elif emotion <= -0.5:
            emotions['anger'] += 1
            emotions['fear'] += 1
            emotions['sadness'] += 1
        elif emotion < 0:
            emotions["anger"] += abs(emotion)
            emotions["sadness"] +=abs(emotion)
            emotions["fear"] += abs(emotion)
    return max(emotions, key=emotions.get)

# Apply emotion detection to processed comment text column
df['emotion'] = df['processed_comment'].apply(lambda x: get_emotion(x))
print(df)

df.to_csv('Comments_Emotional_Sentiment_Analysis.csv', index=False)

# Plot emotion bar chart
emotions_count = df['emotion'].value_counts()
plt.bar(emotions_count.index, emotions_count.values)
plt.show()