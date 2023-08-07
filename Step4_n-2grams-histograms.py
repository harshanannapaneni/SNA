import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('comments_with_2_grams.csv')

# Function to extract the before and next words
def extract_context_words(text, target_word):
    context_words = []
    for _, row in text.iterrows():
        if isinstance(row['2_grams'], str):
            words = row['2_grams'].split(',')
            if len(words) >= 3 and target_word in words[0]:
                before_word, next_word = words[1:3]
                context_words.append((before_word, next_word))
    return context_words

# Target words
target_words = ['investors','FDIC', 'startup', 'government', 'deposit', 'bailout']

# Retrieve before and next words for each target word
context_words = {}
for word in target_words:
    context_words[word] = extract_context_words(df, word)

# Count frequencies of next words for each target word
next_word_freqs = {}
for word, contexts in context_words.items():
    next_words = [next_word for _, next_word in contexts]
    if next_words:
        next_word_freqs[word] = pd.Series(next_words).value_counts().head(10)

# Plot histograms for the more frequent words
for word, freqs in next_word_freqs.items():
    if not freqs.empty:
        plt.figure(figsize=(10, 6))
        freqs.plot(kind='bar')
        plt.title(f"Top 10 Most Frequent Next Words after '{word}'")
        plt.xlabel('Next Word')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No data available for '{word}'.")