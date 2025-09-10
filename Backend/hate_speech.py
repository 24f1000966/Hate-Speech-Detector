import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load dataset
df = pd.read_csv(r"C:\Users\jenaj\OneDrive\Documents\GitHub\Hate-Speech-Detector\Data\Hate_speechdata.csv", encoding='latin1')


# 2. Keep only relevant columns
df = df[['class', 'tweet']]  # 'class' column = labels, 'tweet' column = text
df.dropna(inplace=True)

# 3. Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 4. Text preprocessing function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # remove URLs
    text = re.sub(r'@\w+|\#', '', text)  # remove mentions/hashtags
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = ' '.join(word for word in text.split() if word not in stop_words)  # remove stopwords
    return text

df['clean_text'] = df['tweet'].apply(clean_text)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['class'], test_size=0.2, random_state=42
)

# 6. TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 7. Train model
model = LogisticRegression(max_iter=200)  # added max_iter to avoid convergence warning
model.fit(X_train_tfidf, y_train)

# 8. Evaluate model
y_pred = model.predict(X_test_tfidf)
print("✅ Model Training Completed!")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Save model and vectorizer
joblib.dump(model, "./hate_speech_model.pkl")
joblib.dump(vectorizer, "./vectorizer.pkl")

print("✅ Model and Vectorizer saved successfully!")

# 10. Function for single text prediction
def predict_text(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return "Hate Speech" if model.predict(vec)[0] == 1 else "Not Hate Speech"

# Example test
print(predict_text("Go to hell!"))
print(predict_text("You are amazing!"))

