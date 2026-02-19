import pandas as pd

# NLP + ML utilities
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==================================================
# 1. LOAD RAW TRANSACTION DATA
# ==================================================
# Reads the original transaction data.
# IMPORTANT: Raw data is never modified directly.
# All transformations are done on copies or new columns.

df = pd.read_csv("../data/raw/transactions.csv")

print("Total rows in dataset:", len(df))


# ==================================================
# 2. SPLIT DATA INTO LABELED AND UNLABELED
# ==================================================
# labeled_df  -> rows where human has assigned a category
# unlabeled_df -> rows where category is missing (to be predicted by ML)

labeled_df = df.dropna(subset=["category"])
unlabeled_df = df[df["category"].isna()].copy()

print("Human-labeled rows:", len(labeled_df))
print("Unlabeled rows     :", len(unlabeled_df))


# ==================================================
# 3. DEFINE INPUT (X) AND OUTPUT (y)
# ==================================================
# input_descriptions -> text data given to the ML model
# output_categories  -> correct labels used for supervised learning

input_descriptions = labeled_df["description"]
output_categories = labeled_df["category"]


# ==================================================
# 4. TEXT TO NUMERIC FEATURE CONVERSION (TF-IDF)
# ==================================================
# TF-IDF converts raw text into numerical vectors
# ngram_range=(1,2) -> learns single words and word pairs
# min_df=2          -> ignores very rare words (noise reduction)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2
)

# Learn vocabulary + transform labeled text into numbers
descriptions_tfidf = vectorizer.fit_transform(input_descriptions)

print("TF-IDF feature matrix shape:", descriptions_tfidf.shape)


# ==================================================
# 5. TRAIN / TEST SPLIT
# ==================================================
# Splits labeled data into:
# - Training set (model learns from this)
# - Test set (model is evaluated on unseen data)
#
# stratify=output_categories ensures all categories
# appear proportionally in both sets

X_train, X_test, y_train, y_test = train_test_split(
    descriptions_tfidf,
    output_categories,
    test_size=0.25,
    random_state=42,
    stratify=output_categories
)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ==================================================
# 6. MODEL INITIALIZATION & TRAINING
# ==================================================
# Logistic Regression is used as a strong baseline
# class_weight="balanced" handles class imbalance
# max_iter=1000 ensures model convergence

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# Train the model on labeled training data
model.fit(X_train, y_train)


# ==================================================
# 7. MODEL EVALUATION
# ==================================================
# Evaluate performance on unseen test data

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report (per-category performance):\n")
print(classification_report(y_test, y_pred))


# Confusion matrix shows where the model gets confused
cm = confusion_matrix(y_test, y_pred)
labels = model.classes_

cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print("\nConfusion Matrix:\n")
print(cm_df)


# ==================================================
# 8. PREDICT CATEGORIES FOR UNLABELED TRANSACTIONS
# ==================================================
# IMPORTANT:
# - We use transform(), NOT fit_transform()
# - This ensures the same vocabulary is reused

unlabeled_tfidf = vectorizer.transform(unlabeled_df["description"])

# Predict categories for unlabeled rows
unlabeled_predictions = model.predict(unlabeled_tfidf)

# Store predictions in a separate column
unlabeled_df["predicted_category"] = unlabeled_predictions

print("\nSample ML predictions on unlabeled data:\n")
print(unlabeled_df[["description", "predicted_category"]].head(10))


# ==================================================
# 9. MERGE HUMAN LABELS AND ML PREDICTIONS
# ==================================================
# final_category column is created for downstream use
# Priority:
# - Human label (if exists)
# - ML prediction (if category was missing)

df["final_category"] = df["category"]

df.loc[
    df["final_category"].isna(),
    "final_category"
] = unlabeled_df["predicted_category"]

print("\nFinal category sanity check:\n")
print(df[["description", "category", "final_category"]].head(20))


# ==================================================
# 10. SAVE FINAL PROCESSED DATASET
# ==================================================
# The processed dataset is saved separately
# Raw data remains untouched

df.to_csv(
    "../data/processed/transactions_final.csv",
    index=False
)

print("\nSaved processed data to: data/processed/transactions_final.csv")
