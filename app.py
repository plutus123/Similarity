from flask import Flask, render_template, request, jsonify
import spacy
from spacy.cli.download import download
download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")


app=Flask(__name__)

def semantic_similarity(w1, w2):
    doc1 = nlp(w1)
    doc2 = nlp(w2)
    similarity = doc1.similarity(doc2)
    return similarity


@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict_text():
    sentence1 = request.form.get('sentence1')
    sentence2 = request.form.get('sentence2')
    sim = semantic_similarity(sentence1, sentence2)
    sim = f"Similarity score: {sim}"
    return jsonify({'similarity': str(sim)})


if __name__ == '__main__':
    app.run(debug=True)