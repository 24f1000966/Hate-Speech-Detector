import os
import joblib
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model + vectorizer
model = joblib.load(os.path.join(BASE_DIR, "hate_speech_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

# Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Home route → serve index.html
@app.route("/")
def home():
    return render_template("index.html")

# Predict route → return JSON
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Empty input"}), 400

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    return jsonify({"prediction": str(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
