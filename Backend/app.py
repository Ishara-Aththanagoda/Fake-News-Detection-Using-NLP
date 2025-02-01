from flask import Flask, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")

app = Flask(__name__)

# Load trained models
rf_model = pickle.load(open("random_forest_model.pkl", "rb"))  # Random Forest
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))  # TF-IDF Vectorizer
lstm_model = load_model("lstm_fake_news_model.h5")  # LSTM Model


MAX_SEQUENCE_LENGTH = 500  

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenization
    words = [word for word in words if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(words)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Preprocess text
        cleaned_text = preprocess_text(text)

        # Convert text to TF-IDF features
        text_tfidf = vectorizer.transform([cleaned_text]).toarray()

        # Predict
        rf_prediction = rf_model.predict(text_tfidf)[0]
        rf_result = "Real" if rf_prediction == 1 else "Fake"

        # Convert text to LSTM-compatible
        text_padded = pad_sequences(vectorizer.transform([cleaned_text]).toarray(), maxlen=MAX_SEQUENCE_LENGTH)

        # Predict with LSTM
        lstm_prediction = lstm_model.predict(text_padded)[0][0]
        lstm_result = "Real" if lstm_prediction > 0.5 else "Fake"

        return jsonify({
            "Random_Forest_Prediction": rf_result,
            "LSTM_Prediction": lstm_result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
