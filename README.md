# CSL 7640 – Natural Language Understanding  
## Assignment 1 – Problem 4  
### Sports vs Politics Text Classification

**Author:** Kunal Mishra  
**Roll Number:** M25CSA036  

---

## 📌 Problem Description

This project implements a machine learning–based text classifier that reads a news article and classifies it into one of two categories: **Sports** or **Politics**. The task is a binary text classification problem and is part of Assignment 1 for the course *CSL 7640 – Natural Language Understanding*.

As required in the assignment, the system uses appropriate feature representation techniques and compares the performance of **three different machine learning algorithms**.

---

## 📂 Dataset

The dataset used in this project is derived from the **BBC News dataset**, which contains real-world news articles collected from the British Broadcasting Corporation (BBC).

For this task, only the following categories were used:

- **sport/** – News articles related to sports  
- **politics/** – News articles related to politics  

Each file inside these folders represents a single news article. The dataset consists of approximately **928 documents** in total.

---

## ⚙️ Feature Representation

Text documents are converted into numerical features using the **TF-IDF (Term Frequency–Inverse Document Frequency)** representation.  
Only **unigrams** are used, and the feature space is limited to a fixed size to reduce dimensionality and encourage generalization.

To evaluate model robustness under realistic conditions, **controlled noise** is introduced by randomly removing a fraction of words from each document during preprocessing.

---

## 🤖 Machine Learning Models Used

The following three machine learning classifiers are implemented and compared:

1. **Multinomial Naive Bayes**
2. **Logistic Regression**
3. **Linear Support Vector Machine (SVM)**

All models are trained and evaluated using the same dataset and feature representation to ensure a fair comparison.

---

## 🧪 Experimental Setup

- Training data: **60%**
- Testing data: **40%**
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score

A robustness-oriented setup is used by reducing training data size and introducing controlled noise to simulate real-world text imperfections.

---

## 📊 Results Summary

All three models achieve strong performance with accuracies close to **99%**, even under constrained and noisy conditions.  
Minor variations are observed among the classifiers, reflecting differences in their learning mechanisms and sensitivity to noise.

A detailed quantitative comparison and analysis of results is provided in the accompanying report.

---

## ▶️ How to Run the Code

### Requirements
- Python 3.x
- scikit-learn

### Directory Structure

