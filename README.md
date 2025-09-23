# Detecting Phishing Websites Using Lightweight Machine Learning Models for Enhanced Web Security

## Overview
This project focuses on developing **lightweight machine learning models** for detecting phishing websites in **Kenya’s digital ecosystem**.  
The system is designed to operate efficiently on **resource-constrained devices** (e.g., mobile phones with limited processing power) while maintaining high detection accuracy for **locally targeted phishing threats** (e.g., M-PESA, eCitizen, KRA, banking portals, e-commerce sites).

The project integrates **data acquisition, feature engineering, model development, evaluation, and deployment** into a research-driven prototype.

---

## Objectives
- Analyse Kenya-specific phishing attacks and identify unique patterns.  
- Develop lightweight ML models optimized for phishing detection.  
- Implement a prototype system (API + Browser Extension).  
- Ensure models perform well under **computational and bandwidth constraints**.  
- Benchmark results against existing global phishing detection approaches.  

---

## Project Structure
phishing-detection-kenya/
│
├── data/                     # datasets (global + Kenya-specific)
│   ├── raw/                  # original datasets (PhishTank, OpenPhish, M-PESA phishing, etc.)
│   ├── processed/            # cleaned and preprocessed data
│
├── notebooks/                # Jupyter/Colab notebooks for experiments
│   ├── data_exploration.ipynb
│   ├── feature_engineering.ipynb
│
├── src/                      # source code
│   ├── __init__.py
│   ├── data_processing.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   ├── evaluation.py
│
├── models/                   # saved ML models
│
├── deployment/               # web/API/browser extension prototype
│   ├── api/                  # Flask/FastAPI backend
│   ├── extension/            # Chrome extension code
│
├── docs/                     # documentation (proposal, diagrams, etc.)
│
├── tests/                    # unit & integration tests
│
├── requirements.txt          # Python dependencies
├── README.md
├── .gitignore

---

## Tech Stack
- **Python** (primary ML language)  
- **Scikit-learn**, **XGBoost**, **Optuna** (ML models & optimization)  
- **TensorFlow / TensorFlow.js** (lightweight neural networks & browser deployment)  
- **Flask / FastAPI** (backend APIs)  
- **PostgreSQL** (database for feature storage & metadata)  
- **JavaScript + Chrome Extension APIs** (browser integration)  
- **Google Colab** (experiments & GPU support)  
- **Git & GitHub** (version control & collaboration)  

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/phishing-detection-kenya.git
cd phishing-detection-kenya
```
### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run Jupyter/Colab notebooks
```bash
cd deployment/api
uvicorn main:app --reload
```

---

## Datasets
- Global phishing datasets: PhishTank, Kaggle.
- Kenya-specific phishing URLs:
- M-PESA & Safaricom clones
- KRA & eCitizen portals
- Kenyan banks (Equity, KCB, NCBA)
- Local e-commerce (Jumia, Kilimall)

---

## 📈 Evaluation Metrics
- **Accuracy, Precision, Recall, F1-score**  
- **AUC-ROC** – probabilistic output quality  
- **Latency** – real-time classification within ~5 seconds  
- **Resource usage** – CPU & memory footprint for browser deployment  

---

## 🛠️ Roadmap
- [ ] Collect & preprocess datasets  
- [ ] Feature engineering (URL, content, and Kenya-specific indicators)  
- [ ] Train baseline ML models  
- [ ] Optimize lightweight models (XGBoost, compressed neural networks)  
- [ ] Develop REST API (Flask/FastAPI)  
- [ ] Build browser extension (Chrome APIs + TensorFlow.js)  
- [ ] Deploy and evaluate in real-world scenarios  

---

## 📜 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  

---

## 👨‍💻 Author
**Nathaniel Noel Shibadu**  
---

## 🙌 Acknowledgements
- Open-source contributors and datasets: PhishTank, OpenPhish, Kaggle, Alexa/Tranco  
