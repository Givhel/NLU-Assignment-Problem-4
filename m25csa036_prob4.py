# ------------------------------------------------------------
# CSL 7640 : Natural Language Understanding
# Assignment 1 - Problem 4
# Sports vs Politics Text Classification (Robustness Evaluation)
# Roll Number: M25CSA036
# ------------------------------------------------------------

import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


def add_heavy_noise(text, drop_prob=0.3):
    """
    Randomly removes words from text to simulate noisy real-world data.
    """
    words = text.split()
    noisy_words = [w for w in words if random.random() > drop_prob]
    return " ".join(noisy_words)


def load_dataset():
    """
    Loads BBC News dataset from 'sport/' and 'politics/' folders.
    Each file is treated as one document.
    """
    texts = []
    labels = []

    for filename in os.listdir("sport"):
        file_path = os.path.join("sport", filename)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="latin-1") as f:
                texts.append(add_heavy_noise(f.read()))
                labels.append("sport")

    for filename in os.listdir("politics"):
        file_path = os.path.join("politics", filename)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="latin-1") as f:
                texts.append(add_heavy_noise(f.read()))
                labels.append("politics")

    return texts, labels


def main():
    print("Loading BBC News dataset (robustness setup)...")
    X, y = load_dataset()

    # Reduced training size for robustness evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.6,
        random_state=11
    )

    # TF-IDF feature extraction (unigrams only)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        max_features=2000
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=300),
        "Linear SVM": LinearSVC()
    }

    for name, model in models.items():
        print("\n==============================")
        print("Model:", name)

        model.fit(X_train_vec, y_train)
        predictions = model.predict(X_test_vec)

        print("Accuracy:", accuracy_score(y_test, predictions))
        print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()
