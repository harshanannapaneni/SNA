import pandas as pd
from nltk import ngrams
from wordcloud import WordCloud

df = pd.read_csv('Comments_Data.csv')

def generate_ngrams(text, n):
    n_grams = ngrams(text.split(), n)
    return [' '.join(grams) for grams in n_grams]

df['4_grams'] = df['comment_body'].apply(lambda x: generate_ngrams(x, 4))
df.to_csv('comments_with_4_grams.csv', index=False)

import pandas as pd

# Concatenate all 4-grams into a single string
all_4grams = ' '.join(df['4_grams'].sum())

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_4grams)

# Display the word cloud
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()