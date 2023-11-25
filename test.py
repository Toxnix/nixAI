import json
import pickle
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from os.path import exists

def train_and_save_model(data_path, model_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if 'sentences' in data and isinstance(data['sentences'], list):
        sentences, labels = zip(*[(sentence, label) for sentence, label in data['sentences']])
    else:
        print("Die JSON-Datei hat nicht die erwartete Struktur.")
        return

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    model = MultinomialNB()
    model.fit(X, labels)

    with open(model_path, 'wb') as model_file:
        pickle.dump((vectorizer, model), model_file)

def load_model(model_path):
    if exists(model_path):
        with open(model_path, 'rb') as model_file:
            vectorizer, model = pickle.load(model_file)
        return vectorizer, model
    else:
        print("Kein gespeichertes Modell gefunden.")
        return None, None

def predict_sentence(model, vectorizer, sentence):
    X = vectorizer.transform([sentence])
    prediction = model.predict(X)
    return prediction[0]

# Argument Parser Setup
parser = argparse.ArgumentParser(description='KI-Modell zur Satzanalyse.')
parser.add_argument('sentence', type=str, help='Der zu analysierende Satz.')
args = parser.parse_args()

# Pfade
data_path = '../sample.json'
model_path = 'model.pkl'

# Trainieren und Speichern des Modells
train_and_save_model(data_path, model_path)

# Laden des Modells für Vorhersagen
vectorizer, model = load_model(model_path)
if model:
    sentence = args.sentence
    prediction = predict_sentence(model, vectorizer, sentence)
    print(f"Vorhersage für den Satz: '{sentence}' ist {prediction}")
