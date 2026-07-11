import os
import pickle
import string

import nltk
from flask import Flask, jsonify, render_template, request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Make sure required NLTK data is present (no-op if already downloaded).
for pkg in ('punkt', 'punkt_tab', 'stopwords'):
    try:
        nltk.data.find(f'tokenizers/{pkg}' if 'punkt' in pkg else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))  # load once, not on every request

app = Flask(__name__)

tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))
model = pickle.load(open('model1.pkl', 'rb'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    tokens = [t for t in text if t.isalnum()]
    tokens = [t for t in tokens if t not in STOPWORDS and t not in string.punctuation]
    tokens = [ps.stem(t) for t in tokens]

    return " ".join(tokens)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    msg = (data.get('message') or '').strip()

    if not msg:
        return jsonify({'error': 'Please enter a message.'}), 400

    if len(msg) > 5000:
        return jsonify({'error': 'Message is too long (max 5000 characters).'}), 400

    transformed = transform_text(msg)

    if not transformed:
        # Message had no meaningful tokens after cleaning (e.g. only emoji/punctuation)
        return jsonify({'error': 'Could not extract any usable text from that message.'}), 400

    vector_input = tfidf.transform([transformed])
    result = int(model.predict(vector_input)[0])

    response = {'result': 'Spam' if result == 1 else 'Not Spam'}

    # Surface a confidence score when the underlying model supports it
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vector_input)[0]
        response['confidence'] = float(proba[result])
    elif hasattr(model, 'decision_function'):
        import math
        score = float(model.decision_function(vector_input)[0])
        response['confidence'] = 1 / (1 + math.exp(-abs(score)))  # squashed pseudo-confidence

    return jsonify(response)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)