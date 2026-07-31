# 📞 Telecom Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to **churn** (leave the service) based on their account and service usage data. The pipeline covers data ingestion, preprocessing, class-imbalance handling, model training, and evaluation — with a CI workflow via GitHub Actions.

---

## 📌 Overview

Customer churn is one of the most critical metrics for telecom companies, since retaining an existing customer is far cheaper than acquiring a new one. This project builds a **Random Forest classifier** to identify customers at risk of churning, so that retention strategies can be applied proactively.

---

## 🗂️ Project Structure

```
telecom_churnmodel_prediction/
├── .github/workflows/       # CI pipeline (GitHub Actions)
│   └── python-app.yml
├── data/
│   └── churn.csv            # Telecom customer churn dataset
├── models/
│   └── model.pkl            # Trained model (generated after running)
├── research/
│   └── model.ipynb          # Exploratory notebook / experimentation
├── src/
│   ├── data_ingestion.py    # Loads the dataset
│   ├── data_preprocessing.py# Cleans, encodes, splits & balances data
│   └── model_building.py    # Trains, evaluates & saves the model
├── main.py                  # Entry point that runs the full pipeline
├── requirements.txt         # Project dependencies
└── README.md
```

---

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset (`data/churn.csv`), which contains **7,043 customer records** with **21 features**, including:

- **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account info**: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Services subscribed**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Target variable**: `Churn` (Yes/No)

---

## ⚙️ Pipeline

The workflow is broken into three modular stages, orchestrated by `main.py`:

1. **Data Ingestion** (`src/data_ingestion.py`)
   Loads the churn dataset directly from the repository's raw CSV file into a pandas DataFrame.

2. **Data Preprocessing** (`src/data_preprocessing.py`)
   - Drops duplicate records
   - Maps the target column (`Churn`: Yes → 1, No → 0)
   - Label-encodes categorical features
   - Drops the non-predictive `customerID` column
   - Splits data into train/test sets (70/30 split)
   - Handles **class imbalance** using **SMOTE** (Synthetic Minority Over-sampling Technique) on the training set

3. **Model Building & Evaluation** (`src/model_building.py`)
   - Trains a **Random Forest Classifier**
   - Evaluates performance using **accuracy** and a full **classification report** (precision, recall, F1-score)
   - Saves the trained model as a pickle file to `models/model.pkl`

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+

### Installation

```bash
# Clone the repository
git clone https://github.com/ManojK1104/telecom_churnmodel_prediction.git
cd telecom_churnmodel_prediction

# Install dependencies
pip install -r requirements.txt
```

### Usage

Run the full pipeline (ingestion → preprocessing → training → evaluation):

```bash
python main.py
```

This will print the dataset shape, train/test split shapes, model accuracy, and classification report, then save the trained model to `models/model.pkl`.

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, FLAML (AutoML) |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Environment | Jupyter (ipykernel) |
| CI/CD | GitHub Actions |

---

## 🔄 Continuous Integration

This repo includes a GitHub Actions workflow (`.github/workflows/python-app.yml`) that automatically, on every push/PR to `main`:
- Sets up Python 3.10
- Installs dependencies
- Lints the code with `flake8`
- Runs the pipeline via `python main.py`

---

## 📈 Results

The trained Random Forest model's accuracy and detailed classification metrics (precision, recall, F1-score) are printed to the console at the end of each run of `main.py`. *(Add your latest numbers here once you have a final run, e.g. "Accuracy: 0.79")*

---

## 🛣️ Future Improvements

- Hyperparameter tuning (e.g., GridSearchCV / FLAML AutoML)
- Feature importance / SHAP-based explainability
- Model comparison (Logistic Regression, XGBoost, etc.)
- Deployment via a simple API (Flask/FastAPI) or Streamlit dashboard

---

## 📄 License

This project is licensed under the **Apache-2.0 License** — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

**Manoj K**
GitHub: [@ManojK1104](https://github.com/ManojK1104)
