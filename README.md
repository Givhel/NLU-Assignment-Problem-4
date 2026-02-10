# Sports vs Politics Text Classification

This repository contains the solution for **Problem 4** of the  
**CSL 7640 – Natural Language Understanding** assignment.

The objective is to build a machine learning–based text classifier that can automatically classify a given document as either **Sports** or **Politics**.

---

## Problem Statement

Text classification is a core task in Natural Language Processing (NLP).  
In this problem, a classifier is designed to distinguish between sports-related and politics-related text documents using machine learning techniques.

The system includes:
- Text preprocessing
- Feature extraction
- Training and comparison of multiple classifiers
- Quantitative evaluation of results

---

## Dataset

A **custom dataset** was created for this task.

- The dataset contains short, news-style text documents.
- Each document belongs to one of the following categories:
  - **Sports**
  - **Politics**
- Each document is stored on a separate line.

### Dataset Files
- `sport.txt` – contains sports-related documents
- `politics.txt` – contains politics-related documents

Although the dataset is small, it is sufficient to demonstrate and compare the performance of different machine learning models for text classification.

---

## Feature Representation

Text documents are transformed into numerical features using:

- **TF-IDF (Term Frequency–Inverse Document Frequency)**
- **Unigrams and Bigrams (n-grams)**

This representation helps capture both word importance and short contextual patterns.

---

## Machine Learning Models

The following **three machine learning techniques** were implemented and compared:

1. Multinomial Naive Bayes  
2. Logistic Regression  
3. Linear Support Vector Machine (SVM)

These models are widely used for text classification and provide a good basis for comparison.

---

## Evaluation Metrics

The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

The comparative performance of these models is discussed in detail in the project report.

---

## Files Included

- `M25CSA036_prob4.py` – Python implementation of the classifier
- `sport.txt` – Sports dataset
- `politics.txt` – Politics dataset
- `report.pdf` – Detailed project report
- `README.md` – Project description and instructions

---

## How to Run

1. Ensure the following files are present in the same directory:
   - `M25CSA036_prob4.py`
   - `sport.txt`
   - `politics.txt`

2. Run the program using:
   ```bash
   python M25CSA036_prob4.py

3. The program will train all three models and display their evaluation results.


**Author**
**Course: CSL 7640- Natural Language Understanding**

