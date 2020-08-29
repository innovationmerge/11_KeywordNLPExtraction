from rake_nltk import Rake
import io
import numpy as np
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import matplotlib.pyplot as plt

# Read Text from File
with io.open("DhoniDescription.txt", 'r', encoding='utf8') as f:
    text = f.read().encode('ascii', 'ignore').decode('utf-8')


# Extract keywords from Text using NLP
r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
r.extract_keywords_from_text(text)
extract_keywords = r.get_ranked_phrases()
print(extract_keywords)# To get keyword phrases ranked highest to lowest.
extracted_keywords = '' 
extracted_keywords += " ".join(extract_keywords)+" "

# Stop words
stopwords = set(STOPWORDS) 

# Generate wordcloud
wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='navy', colormap='rainbow', collocations=False, stopwords = STOPWORDS).generate(extracted_keywords)

# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


