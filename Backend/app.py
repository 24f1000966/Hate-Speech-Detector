# from flask import Flask, render_template, request, jsonify
# import joblib

# # Load model and vectorizer
# model = joblib.load("hate_speech_model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# # Flask app
# app = Flask(__name__,
#             template_folder="../frontend/templates",
#             static_folder="../frontend/static")

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
#     text = data.get("text", "")
#     prediction = "❌ Invalid Input"

#     if text.strip():
#         transformed = vectorizer.transform([text])
#         result = model.predict(transformed)[0]
#         prediction = "⚠️ Hate Speech" if result == 1 else "✅ Not Hate Speech"

#     return jsonify({"prediction": prediction})


# # if __name__ == "__main__":
# #     app.run(debug=True)
# if __name__ == "__main__":
#     import os
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port, debug=False)



import os
import joblib
from flask import Flask, render_template, request, jsonify

# Get absolute path of this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and vectorizer with absolute paths
model = joblib.load(os.path.join(BASE_DIR, "hate_speech_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

# Flask app
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)
# @app.route("/")
# def home():
#     return "✅ Hate Speech Detector backend is running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
@app.route("/")
def home():
    return render_template("index.html")
