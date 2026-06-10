# 🏥 Blockchain-Secured Healthcare Provider Fraud Detection
### Using Machine Learning and Explainable AI | INDISCON 2026

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Blockchain](https://img.shields.io/badge/Blockchain-Ethereum%20Ganache-orange)
![ML](https://img.shields.io/badge/ML-7%20Models%20Evaluated-green)
![AUC](https://img.shields.io/badge/AUC--ROC-96.75%25-brightgreen)

---

## 📌 Overview

A full end-to-end machine learning system that detects fraudulent Medicare healthcare providers and permanently records every verdict on an Ethereum blockchain for tamper-proof auditing. Fraud decisions are also explained using SHAP (SHapley Additive exPlanations) so investigators understand *why* a provider was flagged.

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| Best Model | Logistic Regression |
| AUC-ROC | **96.75%** |
| Recall (Fraud Detection Rate) | **90.10%** |
| Overfit Gap | **-0.89%** (best generalisation) |
| Blockchain Records | **134 on-chain, 100% integrity** |
| Features Engineered | **47** (from 3 data sources) |
| Models Evaluated | **7** (LR, RF, SVM, KNN, DT, GB, NB) |

---

## 🗂️ Project Structure

```
healthcare-fraud-detection/
│
├── dashboard/
│   └── appganacheg.py              # Main Streamlit dashboard (9 pages)
│
├── data/
│   ├── raw/                        # Original CMS Medicare CSVs (not uploaded - too large)
│   └── processed/
│       ├── feature_names.csv       # 47 engineered feature names
│       ├── X_val_scaled.npy        # Scaled validation features
│       ├── y_val.npy               # Validation labels
│       ├── lr_shap_values_val.npy  # SHAP values for Logistic Regression
│       ├── rf_shap_values_val.npy  # SHAP values for Random Forest
│       ├── lr_expected_value.npy   # SHAP base value (LR)
│       └── rf_expected_value.npy   # SHAP base value (RF)
│
├── models/
│   ├── logistic_regression.pkl     # Trained LR model
│   ├── random_forest.pkl           # Trained RF model
│   └── scaler.pkl                  # StandardScaler fitted on training data
│
├── results/
│   ├── test_predictions.csv        # ML predictions on 1,353 test providers
│   ├── all_metrics_for_dashboard.csv  # All 7 model metrics
│   ├── optimal_threshold.json      # Tuned classification threshold
│   ├── blockchain_verified_records.csv  # On-chain records
│   ├── blockchain_stored_records.csv    # TxHash + BlockNumber records
│   └── blockchain_config.json      # Deployed contract address + ABI
│
├── blockchain/
│   └── contracts/
│       └── FraudRegistry.sol       # Solidity smart contract
│
├── paper/
│   └── figures/                    # All paper figures (fig06 - fig14)
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_blockchain_store.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔬 Dataset

- **Source:** CMS Medicare Part B Provider Utilization and Payment Data
- **Training providers:** 5,410 (506 fraud = 9.4%, 4,904 clean = 90.6%)
- **Test providers:** 1,353
- **Imbalance handling:** SMOTE (balanced to 3,923 each class → 7,846 total)
- **Three raw data sources:** Beneficiary demographics, Inpatient claims, Outpatient claims

---

## ⚙️ Feature Engineering (47 Features)

Features are aggregated at the **provider level** (each row = one healthcare provider):

| Category | Count | Examples |
|----------|-------|---------|
| Inpatient Financial | 8 | IP_TotalClaimAmt, IP_AvgClaimAmt, IP_MaxClaimAmt |
| Inpatient Behavioral | 8 | IP_NumClaims, IP_NumUniqueAttPhysicians |
| Inpatient Duration | 4 | IP_AvgStayDuration, IP_TotalStayDays |
| Inpatient Patient | 6 | IP_AvgPatientAge, IP_PctDeadPatients |
| Outpatient | 13 | OP_NumClaims, OP_TotalClaimAmt, OP_AvgDiagnosisCodes |
| Cross-Domain Ratios | 8 | IP_OP_ClaimRatio, Total_ClaimAmt, DiagComplexityRatio |

---

## 🤖 ML Pipeline

```
Raw CSVs → Feature Engineering (47 features) → SMOTE → 
StandardScaler → 7 Models → Threshold Tuning → SHAP Explainability
```

**Models evaluated:** Logistic Regression, Random Forest, SVM, KNN, Decision Tree, Gradient Boosting, Naive Bayes

**Why Logistic Regression was chosen:**
- Highest AUC-ROC (96.75%)
- Best Overfit Gap (-0.89%) → best generalisation
- Highest Recall (90.10%) → catches most fraud
- Fastest inference (0.002s per batch)

---

## ⛓️ Blockchain Architecture

### Dual-Chain Design

| Chain | Technology | Purpose |
|-------|-----------|---------|
| Ethereum (Ganache) | Solidity + web3.py | Production tamper-proof audit trail |
| SHA-256 In-Memory | Python hashlib | Interactive demo + offline fallback |

### Smart Contract Functions
```solidity
storeFraudRecord(providerID, isFraud, fraudProbability, riskCategory, dataHash)
// Gas: ~170,412 per record | Time: 0.151s

getRecord(uint256 recordId) → (string, bool, uint256, string, uint256, bytes32)
// Gas: 0 (view function) | Time: 0.075s

verifyRecord(recordId, expectedHash) → bool
// Tamper detection
```

### Tamper Detection
Each record is hashed:
```
SHA-256(providerID + isFraud + probability(6dp) + riskCategory) = block_hash
```
Each block stores the previous block's hash → breaking any record breaks the entire chain.

---

## 🖥️ Dashboard Pages

| Page | What It Shows |
|------|--------------|
| 📊 Overview | KPIs, risk distribution, system architecture |
| 📤 Upload & Predict | Upload new CMS CSVs → instant fraud predictions |
| 📈 Model Performance | All 7 models compared, ROC curves, confusion matrices |
| 🔍 Fraud Analysis | Filter/search providers, individual probability gauge |
| 🧠 Explainable AI | SHAP global + local explanations, LR vs RF comparison |
| 🔐 Blockchain Security | Live tamper demo with SHA-256 chain verification |
| ⛓️ Blockchain Records | SHA-256 in-memory ledger with integrity check |
| ⛓️ Blockchain Ledger | Live Ganache Ethereum records (auto-refresh 15s) |
| ℹ️ System Info | Dataset stats, runtime benchmarks, paper metadata |

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/healthcare-fraud-detection.git
cd healthcare-fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Ganache (for blockchain features)
- Download [Ganache](https://trufflesuite.com/ganache/)
- Open your saved Workspace or start a new one on port **7545**
- Make sure your deployed contract address matches `results/blockchain_config.json`

### 4. Run the dashboard
```bash
streamlit run dashboard/appganacheg.py
```

### 5. Open in browser
```
http://localhost:8501
```

---

## 📦 Requirements

See `requirements.txt` for full list. Key packages:
- `streamlit` — dashboard framework
- `scikit-learn` — ML models + SMOTE
- `shap` — explainability
- `web3` — Ethereum blockchain integration
- `imbalanced-learn` — SMOTE oversampling
- `joblib` — model serialisation

---

## 📄 Paper

**Title:** Blockchain-Secured Healthcare Provider Fraud Detection using Machine Learning and Explainable AI

**Conference:** INDISCON 2026

**Abstract:** This paper presents an integrated system combining 47 engineered features from CMS Medicare data, 7 evaluated ML models (best: Logistic Regression, AUC-ROC 96.75%, Recall 90.10%), SHAP-based explainability, and an Ethereum blockchain audit trail with 100% tamper-proof integrity across 134 stored records.

---

## 👩‍💻 Author

**Geetha** | INDISCON 2026
