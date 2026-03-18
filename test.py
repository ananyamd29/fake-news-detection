import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 1️⃣ Load CSV files
fake = pd.read_csv("data/Fake.csv", nrows=500)   # load first 500 rows (faster)
true = pd.read_csv("data/True.csv", nrows=500)

# 2️⃣ Add labels
fake["label"] = "FAKE"
true["label"] = "REAL"

# 3️⃣ Combine datasets
df = pd.concat([fake, true])

# 4️⃣ Combine title + text into one column
df["content"] = df["title"] + " " + df["text"]
df = df[["content", "label"]]

# 5️⃣ Clean the text
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # remove non-letters
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

df["content"] = df["content"].apply(clean_text)

# 6️⃣ Convert text → numbers (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["content"])
y = df["label"]

# 7️⃣ Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Check results
print("First 5 cleaned rows:")
print(df.head())
print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Initialize model
model = LogisticRegression(max_iter=1000)

# 2️⃣ Train model
model.fit(X_train, y_train)

# 3️⃣ Predict on test set
y_pred = model.predict(X_test)

# 4️⃣ Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Test news headline
while True:
    text = input("\nEnter news to test (or type 'exit' to quit): ")
    if text.lower() == "exit":
        break

    # Clean text
    def clean_text_input(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    cleaned = clean_text_input(text)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    print("Prediction:", prediction)