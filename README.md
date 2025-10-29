# ✈️ Twitter Sentiment Analysis on US Airplanes Using Random Forest Classifier

This project performs **multiclass sentiment classification** on tweets about major US airlines to analyze public perception toward airline services. It employs a **Random Forest Classifier** trained on multiple **text representation methods**—Bag of Words (BoW), TF-IDF, Word2Vec, GloVe, and FastText—to compare their effectiveness in modeling textual features. The workflow integrates **10-Fold GridSearch hyperparameter tuning** and evaluates model performance using **weighted and macro F1-scores** on both imbalanced and SMOTE-balanced datasets to identify the most robust representation-model combination for sentiment prediction.

---

## 🔧 Features

- **Exploratory Data Analysis (EDA)** to understand sentiment distribution and dataset characteristics.
- **Text Preprocessing** including:
  - Cleaning (mentions, URLs, emojis, and punctuation removal)
  - Tokenization
  - Stopword removal
- **Text Representation Methods**:
  - Bag of Words (BoW)
  - TF-IDF
  - Word2Vec
  - GloVe (pretrained)
  - FastText (pretrained)
- **Model Training**:
  - Random Forest Classifier
- **Hyperparameter Optimization**:
  - GridSearchCV with 10-Fold Cross-Validation
- **Evaluation Metrics**:
  - Weighted F1-Score on imbalanced data
  - Macro F1-Score on SMOTE-balanced data
  - Classification report for precision, recall, and F1 per class

---

## 🧠 Concepts Used

- **Natural Language Processing (NLP)** for text cleaning and tokenization.  
- **Feature Representation** comparison between frequency-based (BoW, TF-IDF) and embedding-based (Word2Vec, GloVe, FastText) methods.  
- **Ensemble Learning** with Random Forest for text classification.  
- **SMOTE (Synthetic Minority Oversampling Technique)** to address class imbalance.  
- **GridSearchCV with K-Fold Cross Validation (k=10)** for systematic hyperparameter tuning.  
- **F1-Score (Weighted & Macro)** for evaluating performance under different class distributions.
