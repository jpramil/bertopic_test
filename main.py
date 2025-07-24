import os
import s3fs
import pandas as pd

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

import re
import nltk
from nltk.corpus import stopwords

# À exécuter une fois
nltk.download('stopwords')
stopwords_fr = set(stopwords.words('french'))

def preprocess(text):
    # Supprime la ponctuation, met en minuscule
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Supprime les stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords_fr])
    return text

docs = [preprocess(doc) for doc in docs]

import pandas as pd
from io import StringIO
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import tiktoken
import time

model_name = "paraphrase-multilingual-MiniLM-L12-v2"
# model_name = "distiluse-base-multilingual-cased-v2"

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





# Sauvegarde du modèle
topic_model.save("bertopic_model_fr")

# Visualisation (HTML interactive)
fig = topic_model.visualize_topics()
fig.write_html("topics_visualization.html")

# Optionnel : documents par topic
df_resultats = pd.DataFrame({
    "texte": docs,
    "topic": topics,
    "probabilité": probs
})
df_resultats.to_csv("resultats_topic.csv", index=False)