import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Lade die Daten aus der JSON-Datei
with open('sample.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Überprüfe, ob die JSON-Datei das erwartete Format hat
if 'sentences' in data and isinstance(data['sentences'], list):
    sentences_data = data['sentences']
else:
    print("Die JSON-Datei hat nicht die erwartete Struktur.")
    exit(1)

# Extrahiere Sätze, Klassifizierungen und Gründe
sentences = [item['sentence'] for item in sentences_data]
toxic_labels = [item['toxic'] for item in sentences_data]
toxic_reasons = [item.get('toxicReason', '') for item in sentences_data]

# Feature-Extraktion (TF-IDF)
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(sentences)

# Modelltraining (Naive Bayes)
model = MultinomialNB()
model.fit(X, toxic_labels)

# Beispieltext klassifizieren
new_text = "Wie geht es dir, arsch"
new_text_tfidf = tfidf_vectorizer.transform([new_text])
predicted_toxic = model.predict(new_text_tfidf)

# Erstelle die Ausgabe im JSON-Format
output = {
    "sentence": new_text,
    "toxic": bool(predicted_toxic[0]),
    "toxicReason": toxic_reasons[0] if predicted_toxic[0] else ""
}

print(json.dumps(output, ensure_ascii=False, indent=4))
