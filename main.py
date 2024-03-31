import pandas as pd
import spacy
data=pd.read_csv("DataNeuron_Text_Similarity.csv")

w1=data["text1"][0]
w2=data["text2"][0]

nlp=spacy.load("en_core_web_lg")

def semantic_similarity(w1, w2):
    doc1 = nlp(w1)
    doc2 = nlp(w2)
    similarity = doc1.similarity(doc2)
    return similarity

# similarity = semantic_similarity(w1, w2)
# print(f"Similarity score: {similarity}")



df = data.copy()

df['similarity_score'] = df.apply(lambda row: semantic_similarity(row['text1'], row['text2']), axis=1)

print(df)
