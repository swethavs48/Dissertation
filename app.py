from flask import Flask, render_template, request
from pathlib import Path
from models import ticket_prediction
from models import recommendation
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
# import nltk
# nltk.download('stopwords')
# from nltk.cluster.util import cosine_distance
# import networkx as nx

app = Flask(__name__, template_folder='app/templates',static_folder='static')

df = pd.read_csv(Path('data','support_tickets_history.csv'), low_memory=False)
missing_values = ['NA', 'N/A', 'NULL', 'None', '', ' ', 'Unknown']  # Add other representations if needed
df.replace(missing_values, np.nan, inplace=True)

# Ayra models
ayra_models = {'ticket_prediction_model':Path('data','best_model.sav')}

# Initialize NLP models
# nlp = spacy.load('en_core_web_sm')

# def read_article(text):
#     file = open(Path('data','Medical_Report.txt'), "r")
#     filedata = file.readlines()
#     article = filedata[0].split(". ")
#     sentences = []
#     for sentence in article:
#         sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
#     sentences.pop()
#     return sentences

# def sentence_similarity(sent1, sent2, stopwords=None):
#     if stopwords is None:
#         stopwords = []
#     sent1 = [w.lower() for w in sent1]
#     sent2 = [w.lower() for w in sent2]
#     all_words = list(set(sent1+sent2))
#     vector1 = [0]*len(all_words)
#     vector2 = [0]*len(all_words)

#     for w in sent1:
#         if w in stopwords:
#             continue
#         vector1[all_words.index(w)] +=1
#     for w in sent2:
#         if w in stopwords:
#             continue
#         vector2[all_words.index(w)] +=1
#     return 1-cosine_distance(vector1,vector2)

# def gen_sim_matrix(sentences, stopwords):
#     similarity_matrix = np.zeros((len(sentences), len(sentences)))
#     for idx1 in range(len(sentences)):
#         for idx2 in range(len(sentences)):
#             if idx1 == idx2:
#                 continue
#             similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stopwords)
#     return similarity_matrix

# def generate_summary(file_name, top_n=5):
#     stop_words = nltk.corpus
#     summarize_text = []
#     sentences = read_article(file_name)
#     sentence_similarity_matrix = gen_sim_matrix(sentences, stop_words)
#     sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
#     scores = nx.pagerank(sentence_similarity_graph)
#     ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
#     for i in range(top_n):
#         summarize_text.append(" ".join(ranked_sentence[i][1]))
#     print("Summarize Text: \n", ". ".join(summarize_text))

# # generate_summary(Path('data','Medical_Report.txt'), 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendation', methods=['POST'])
def recommend():
    # Get the input text from the form
    input_text = request.form['input_text']
    recommendations = recommendation.recommend_resolution(df, input_text)
    return render_template('index.html', recommendations=recommendations)

@app.route('/ticket_prediction', methods=['POST'])
def ticket_predict():
    # Get the input date from the form
    input_date = request.form['input_date']
    prediction, prediction_graph = ticket_prediction.predict_no_of_tickets_xb(input_date, ayra_models['ticket_prediction_model'])
    return render_template('index.html', prediction=prediction, prediction_graph=prediction_graph)

# @app.route('/summarize_notes', methods=['POST'])
# def summarize_uploaded_notes():
#     if 'uploaded_notes' in request.files:
#         uploaded_notes = request.files['uploaded_notes']
#         notes_text = uploaded_notes.read().decode('utf-8')
        
#         summarizer = pipeline("summarization")
#         summary = summarizer(notes_text, max_length=1000, min_length=200, do_sample=False)
        
#         return jsonify({'notes_summary': summary[0]['summary_text']})
#     else:
#         return jsonify({'error': 'No notes uploaded'})

if __name__ == '__main__':
    app.run(host="127.0.0.1",port=8080,debug=True)
    # generate_summary(Path('data','test.txt'), 2)