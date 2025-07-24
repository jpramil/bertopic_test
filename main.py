import os
import s3fs
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from io import StringIO
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import time

## Import texts ======================

# S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
# fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

# BUCKET = "jpramil"
# FILE_KEY_S3 = "fra_sentences.tsv"
# FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3

# with fs.open(FILE_PATH_S3, mode="rb") as file_in:
#     df = pd.read_csv(file_in, sep='\t', header=None, names=["index", "lang", "sentence"])

df = pd.read_csv("textes_topics.tsv", sep='\t')

df.columns
df.shape
# df_sample = df.sample(n=10)
# docs = df_sample["sentence"].dropna().astype(str).tolist()
docs = df["text"].dropna().astype(str).tolist()


## Preprocessing ================================
## (utile pour les mots représentatifs, par pour la définition des clusters)

nltk.download('stopwords')
stopwords_fr = set(stopwords.words('french'))

def preprocess(text):
    # Supprime la ponctuation, met en minuscule
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Supprime les stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords_fr])
    return text

docs = [preprocess(doc) for doc in docs]



# model_name = "paraphrase-multilingual-MiniLM-L12-v2"
# model_name = "distiluse-base-multilingual-cased-v2"
model_name = "distiluse-base-multilingual-cased-v1"

embedding_model = SentenceTransformer(model_name)

# ---------- 4. BERTopic ----------
topic_model = BERTopic(embedding_model=embedding_model, language="french", calculate_probabilities=True)

start_time = time.time()
topics, probs = topic_model.fit_transform(docs)
total_time = time.time() - start_time
print(f"Total time per document: {round(total_time / len(docs), 2)}")

# ---------- 5. Résultats ----------
# Info sur les topics

df_topics = topic_model.get_topic_info()
print(df_topics)

print(topic_model.get_topic(1))
topic_model.get_document_info(docs)


## Test avec camenbert ==================

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import re
import tiktoken

nltk.download('stopwords')
stopwords_fr = list(stopwords.words("french")) 

def preprocess(doc):
    txt = re.sub(r'[^\w\s]', '', doc.lower())
    return ' '.join(w for w in txt.split() if w not in stopwords_fr)

# Charger le modèle CamemBERT finement ajusté
model = SentenceTransformer("sentence-transformers/LaBSE")
# model = SentenceTransformer("dangvantuan/sentence-camembert-large")
# model = SentenceTransformer("dangvantuan/sentence-camembert-base")

# Prétraiter les documents
docs_clean = [preprocess(doc) for doc in docs]

# Créer embeddings
embeddings = model.encode(docs, show_progress_bar=True)

# Créer vectorizer personnalisé pour TF-IDF des topics
vectorizer = CountVectorizer(stop_words=stopwords_fr)

# Ajuster BERTopic
topic_model = BERTopic(embedding_model=model,
                       vectorizer_model=vectorizer,
                       language="french")
topics, probs = topic_model.fit_transform(docs_clean, embeddings)

# Inspecter les résultats
print(topic_model.get_topic_info())
print(topic_model.get_document_info(docs))
topic_model.visualize_barchart(top_n_topics=5)