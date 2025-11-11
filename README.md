Bust-Out Fraud Detection using Machine Learning

This project focuses on detecting bust-out scams, a type of financial fraud where fraudsters build positive credit history through regular transactions and then “bust out” by maxing their credit line with no intention of repayment.

The solution uses LightGBM with behavioral and temporal feature engineering to improve early fraud detection performance beyond rule-based systems.

Environment Setup

Clone this repository and install all dependencies:

git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt


Alternatively, if using Google Colab, run:

!pip install -r requirements.txt

Dependencies

This project was developed using Python 3.10+.
Below are the main dependencies required:

Library	Purpose
pandas	Data manipulation and aggregation
numpy	Numerical operations
matplotlib	Visualizations and feature importance plots
scikit-learn	Model evaluation, metrics, and utility functions
lightgbm	Gradient boosting classifier for fraud detection
optuna	Hyperparameter optimization
joblib	Saving and loading trained models
pyarrow	For efficient .parquet file handling
tqdm (optional)	Progress tracking
seaborn (optional)	Enhanced visualizations
Example requirements.txt
pandas>=2.0.0
numpy>=1.23.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
lightgbm>=4.1.0
optuna>=3.5.0
joblib>=1.3.0
pyarrow>=14.0.0
tqdm>=4.66.0
seaborn>=0.13.0

How to Reproduce

Mount your Google Drive and load the dataset:

df = pd.read_parquet("/content/drive/MyDrive/mlba project/fraud_user_time_downsampled.parquet")


Run the data split and feature engineering scripts.

Train the model using the tuned hyperparameters.

Evaluate performance (Precision-Recall AUC, F1 threshold).

Save the final model:

joblib.dump(final_model, "/content/drive/MyDrive/mlba project/lightgbm_final_tuned_model.pkl")

Model Overview

Algorithm: LightGBM (Gradient Boosted Trees)

Features: Temporal and behavioral (rolling sums, merchant diversity, transaction gaps)

Metrics: Precision-Recall AUC, F1-score, Precision, Recall

Goal: Detect early bust-out frauds with higher recall and explainability

 Project Structure
├── fraud_detection.ipynb           # Main Colab notebook
├── requirements.txt                # Dependency list
├── README.md                       # Project documentation
├── /data                           # Input parquet files (not pushed to GitHub)
├── /models                         # Saved model files (.pkl)
└── /outputs                        # Evaluation metrics & visualizations

 Results Summary
Metric	Score
Precision-Recall AUC	~0.60
F1-Score	~0.55
Accuracy	~0.98
Recall	~0.51
