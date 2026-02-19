# Z PAY – Expense Categorization (ML)

This repository contains the **machine learning expense categorization engine**
for the **Z PAY UPI application**.

The goal of this module is to automatically categorize transaction descriptions  
(e.g. `ZMT ORD`, `UBR TRIP`, `AMZN MKTP`) into meaningful expense categories such as:

- Food  
- Travel  
- Shopping  
- Subscriptions  
- Entertainment  
- Other  

This project focuses on building a **realistic ML pipeline**, not just training a model.

---

## ✨ What this project does

- Works with **partially labeled transaction data**
- Avoids rule-based categorization
- Trains a machine learning model using human-labeled samples
- Predicts categories for unlabeled transactions
- Preserves human labels and fills missing values using ML
- Produces a final, fully categorized dataset ready for use

---

## 🧠 How the pipeline works

1. Load raw transaction data  
2. Separate labeled and unlabeled transactions  
3. Convert transaction descriptions into numerical features using **TF-IDF**  
4. Train a **Logistic Regression** model on labeled data  
5. Evaluate the model on unseen test data  
6. Predict categories for unlabeled transactions  
7. Merge human labels with ML predictions safely  
8. Save the final processed dataset  

---

## 📊 Results (current)

- Trained using **~70 human-labeled transactions**
- Achieved **~94% accuracy** on a held-out test set (**18 samples**)
- Automatically predicted categories for **~530 unlabeled transactions**

> The focus is on correctness, data handling, and pipeline design  
> rather than maximizing accuracy on a small dataset.

---

## 📁 Folder structure

z-pay-expense-categorization/<br>
│<br>
├── models/<br>
│ └── expense_model.py # ML pipeline code<br>
│<br>
├── data/<br>
│ ├── raw/ # original datasets (untouched)<br>
│ │ ├── categories.csv<br>
│ │ ├── transactions.csv<br>
│ │ └── users.csv<br>
│ │<br>
│ └── processed/ # ML-generated output<br>
│ &emsp;└── transactions_final.csv

---

## 🛠️ Tech stack

- Python  
- Pandas  
- scikit-learn  

---

## 🚧 Project status

✅ Expense categorization pipeline completed  

🔜 Planned extensions:
- Confidence scores for predictions
- Model persistence (save / load)
- Fraud detection module
- Backend & database integration

---

## ℹ️ Note

This repository represents **only the ML component** of the Z PAY project.  
Other components such as backend services, APIs, and frontend applications
will be developed separately.
