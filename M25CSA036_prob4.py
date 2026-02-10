# ------------------------------------------------------------
# CSL 7640 : Natural Language Understanding
# Assignment 1 - Problem 4
# Sports vs Politics Classification
# Roll Number: M25CSA036
# ------------------------------------------------------------

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    texts = []
    labels = []

    with open("sport.txt", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(line.strip())
                labels.append("sport")

    with open("politics.txt", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(line.strip())
                labels.append("politics")

    return texts, labels


def main():
    print("Loading dataset...")
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC()
    }

    for name, model in models.items():
        print("\n==============================")
        print("Model:", name)

        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)

        print("Accuracy:", accuracy_score(y_test, preds))
        print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()
