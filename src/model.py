import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from security.input_validator import is_input_strong


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text


data = pd.read_csv(r"C:\ML Projects\AI Resume Predicter\src\data.csv")

print(data.head())


data["resume"] = data["resume"].apply(clean_text)

X = data["resume"]
Y = data["category"]


X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)


model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])


model.fit(X_Train, Y_Train)


Y_Pred = model.predict(X_Test)

print("\nAccuracy:", accuracy_score(Y_Test, Y_Pred))

print("\nClassification Report:\n")
print(classification_report(Y_Test, Y_Pred))


joblib.dump(model, "resume_classifier.pkl")

print("\nModel saved as resume_classifier.pkl")


print("\n\nMODEL TRAINED SUCCESSFULLY!")
print("NOW STARTING PREDICTION SYSTEM...\n")


while True:
    print("\n==============================")
    print("AI Resume Category Predictor")
    print("==============================")

    user_resume = input("\nEnter Resume Text: ")

    if user_resume.lower() == "exit":
        print("\nProgram Closed")
        break

    is_valid, message = is_input_strong(user_resume)

    if not is_valid:
        print("\n", message)
        continue

    cleaned_resume = clean_text(user_resume)

    prediction = model.predict([cleaned_resume])[0]

    probabilities = model.predict_proba([cleaned_resume])[0]

    confidence = max(probabilities) * 100

    print("\nPredicted Category:", prediction)

    print(f"Confidence Score: {confidence:.2f}%")