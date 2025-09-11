from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data["text"]

    # Transform text and make prediction
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    # User-friendly result
    result = "✅ Not Hate Speech" if prediction == 0 else "❌ Hate Speech"

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
