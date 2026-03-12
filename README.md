# Z PAY -- Intelligent UPI Analytics Platform

**Z PAY** is a machine‑learning powered financial analytics system
designed to enhance traditional UPI transaction systems.

The goal of the project is to build a **smart backend engine** capable
of analyzing transaction data and generating meaningful insights such
as:

• Automatic expense categorization\
• Fraud risk detection\
• Behavioral transaction analysis
Instead of relying on rigid rule‑based systems, Z PAY uses **machine
learning models and data pipelines** to understand patterns in financial
transactions.

The long‑term vision is to build a **modern fintech intelligence layer**
that can sit on top of payment platforms and provide smarter financial
insights.

------------------------------------------------------------------------

# Core Machine Learning Modules

The current system contains two major ML pipelines.

------------------------------------------------------------------------

## 1️⃣ Expense Categorization Model

This model automatically categorizes raw transaction descriptions into
meaningful expense categories.

Example:

ZMT ORD → Food\
UBR TRIP → Travel\
AMZN MKTP → Shopping

### How it works

1.  Raw transaction descriptions are processed
2.  Text is converted into numerical features using **TF‑IDF**
3.  A **Logistic Regression model** learns patterns from human‑labeled
    data
4.  The model predicts categories for unlabeled transactions
5.  Human labels are preserved while missing values are filled using ML

### Achievements

• Trained using **\~70 human‑labeled transactions**\
• Achieved **\~94% accuracy on a held‑out test set**\
• Automatically categorized **\~530 previously unlabeled transactions**

This model forms the **foundation of spending analytics** in the system.

------------------------------------------------------------------------

## 2️⃣ Fraud Detection Model

This module detects suspicious financial transactions using a **hybrid
anomaly detection system**.

Unlike traditional fraud systems that require labeled fraud cases, this
pipeline works even when fraud labels are unavailable.

### How it works

1.  Transaction features are engineered from the dataset
    (amount, time, location, device, category)
2.  **Isolation Forest** detects anomalous transactions
3.  Detected anomalies are used as **pseudo fraud labels**
4.  A **Random Forest classifier** learns patterns from these anomalies
5.  The model generates:\
      • fraud probability scores\
      • fraud flags\
      • human‑readable fraud explanations

Example explanation:

High transaction amount, Late‑night transaction

### Achievements

• Analyzed **\~600 financial transactions**\
• Detected **\~5% anomalous transactions**\
• Generated **fraud risk scores for every transaction**

This allows the system to simulate **real‑world fraud monitoring
pipelines**.

------------------------------------------------------------------------

## 📁 Folder structure

Z_PAY/<br>
├── backend<br>
│  └── app.py \# fastapi backend or real‑time transaction analysis
├── data/<br>
│ ├── raw/ \# original datasets (untouched)<br>
│ │ ├── categories.csv<br>
│ │ ├── transactions.csv<br>
│ │ └── users.csv<br>
│ │<br>
│ └── processed/ \# ML‑generated datasets<br>
│   ├── transactions_final.csv<br>
│   └── transactions_with_fraud.csv<br>
│<br>
├── models/<br>
│ ├── expense_model.py \# expense categorization pipeline<br>
│ └── fraud_detection.py \# fraud detection pipeline<br>
│<br>
├── README.md<br>
└── requirements.txt<br>

------------------------------------------------------------------------

## 🛠️ Tech stack

-   Python
-   Pandas
-   scikit‑learn
-   TF‑IDF (NLP feature extraction)
-   Logistic Regression
-   Isolation Forest
-   Random Forest

------------------------------------------------------------------------

## 🚧 Project status

✅ Expense categorization ML pipeline\
✅ Fraud detection hybrid model\
✅ Transaction analysis dataset generation\
✅ FastAPI backend for real‑time transaction analysis
