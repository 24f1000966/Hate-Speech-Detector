from flask import Flask, render_template, request, jsonify
import joblib

# Load model and vectorizer
model = joblib.load("hate_speech_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Flask app
app = Flask(__name__,
            template_folder="../frontend/templates",
            static_folder="../frontend/static")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    prediction = "❌ Invalid Input"

    if text.strip():
        transformed = vectorizer.transform([text])
        result = model.predict(transformed)[0]
        prediction = "⚠️ Hate Speech" if result == 1 else "✅ Not Hate Speech"

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
