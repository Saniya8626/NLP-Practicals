,# Import Libraries
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec


# Sample Dataset
sentences = [
    "Data science is fun",
    "Data science is interesting"
]

print("\nOriginal Sentences:")
print(sentences)


# ------------------------------------------------
# 1️⃣ Bag of Words (Count Occurrence)
# ------------------------------------------------

print("\nBag of Words (Count Occurrence):")

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(sentences)

print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nCount Matrix:")
print(X.toarray())


# ------------------------------------------------
# 2️⃣ Normalized Count Occurrence
# ------------------------------------------------

print("\nNormalized Count Occurrence:")

count_matrix = X.toarray()

normalized_matrix = count_matrix / count_matrix.sum(axis=1, keepdims=True)

print(normalized_matrix)


# ------------------------------------------------
# 3️⃣ TF-IDF
# ------------------------------------------------

print("\nTF-IDF Representation:")

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(sentences)

print("Vocabulary:")
print(tfidf.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())


# ------------------------------------------------
# 4️⃣ Word2Vec Embeddings
# ------------------------------------------------

print("\nWord2Vec Embeddings:")

# Tokenize sentences
tokenized_sentences = [sentence.split() for sentence in sentences]

# Train Word2Vec Model
w2v_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=50,
    window=2,
    min_count=1,
    workers=1
)

# Get word vector
print("\nWord Vector for 'science':")

print(w2v_model.wv['science'])