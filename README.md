````markdown
# Bust-Out Fraud Detection using Cost-Sensitive Machine Learning

## 📌 Project Overview
This project targets **Bust-Out Fraud** (also known as First-Party Fraud), a scheme where fraudsters cultivate a positive credit history through regular transactions before suddenly "busting out" by maxing their credit lines with no intention of repayment.

Unlike traditional binary classifiers that optimize for accuracy, this solution implements a **Cost-Sensitive Learning** framework. It prioritizes financial impact over raw probability by employing an **Expected Loss Strategy**, ensuring that investigator resources are focused solely on high-value, high-risk transactions.

## 🚀 Key Features
* **Behavioral Feature Engineering:** Velocity tracking (7-day/30-day rolling sums) to catch the "acceleration" phase of bust-out behavior.
* **Cyclical Temporal Encoding:** Transformation of `Hour` and `Month` into Sine/Cosine features to preserve temporal continuity.
* **Leakage Prevention:** Explicit removal of longitudinal features (e.g., `Year`) to prevent overfitting to historical inflation or time-drift.
* **Financial Threshold Optimization:** A custom decision engine that calculates net profitability to determine the optimal alert threshold, rather than using arbitrary probability cutoffs.

## 🛠️ Environment Setup

Clone this repository and install dependencies:

```bash
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
````

**Note:** This project relies on `.parquet` files for efficient data handling. Ensure you have `pyarrow` or `fastparquet` installed.

## 📦 Dependencies

| Library | Purpose |
| :--- | :--- |
| **pandas** | Data manipulation and rolling window feature generation |
| **numpy** | Numerical operations and cyclical encoding |
| **lightgbm** | Gradient boosting classifier (optimized for imbalance) |
| **optuna** | Bayesian hyperparameter optimization |
| **scikit-learn** | Precision-Recall curves and evaluation metrics |
| **joblib** | Model serialization |
| **pyarrow** | High-performance Parquet file I/O |
| **matplotlib/seaborn** | Visualization of Profit Curves and Feature Importance |

## 🔄 How to Reproduce

1.  **Data Loading:**
    Ensure your dataset is in Parquet format. Update paths if not using Google Drive.

    ```python
    df = pd.read_parquet("path/to/fraud_user_time_downsampled.parquet")
    ```

2.  **Feature Engineering:**
    Run the preprocessing pipeline to generate velocity features and apply cyclical encoding. **Crucial:** Ensure the `Year` column is dropped before training.

3.  **Optimization:**
    Run the Optuna study to find the best hyperparameters for `scale_pos_weight` and tree depth.

4.  **Financial Analysis (The Core Logic):**
    The notebook calculates the **Net Savings** curve:
    $$ \text{Savings} = \sum (\text{Caught Fraud Amount}) - (\text{Alerts} \times \text{Admin Cost}) $$

5.  **Inference:**
    Run the final screener which flags transactions based on Risk Score:
    `Risk Score = Probability × Transaction Amount`

## 📊 Model & Results

### Model Configuration

  * **Algorithm:** LightGBM Classifier
  * **Tuning:** Optuna (Bayesian Optimization)
  * **Objective:** Maximize Precision-Recall AUC (initially), optimized for Net Profit (final).

### Financial Performance (Test Set)

Instead of optimizing for abstract metrics like Accuracy, the final model is tuned for **Profitability**.

| Metric | Result |
| :--- | :--- |
| **Optimal Risk Threshold** | **$86.53** (Prob × Amount) |
| **Projected Net Savings** | **~$8,000** (vs doing nothing) |
| **Workload** | Top **0.9%** of transactions flagged |
| **Precision** | **63.5%** (High confidence alerts) |
| **Recall** | **41.6%** (Captures the highest value fraud) |

*Note: While Recall seems low (41.6%), the model successfully ignores low-value fraud (e.g., $5) that costs more to investigate than the loss itself.*

## 📂 Project Structure

```text
├── model.ipynb       # End-to-end pipeline (Preprocessing -> Optuna -> Financial Analysis)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── /data                       # Data folder (Excluded from repo)
├── /models                     # Serialized models
│   └── lightgbm_final.pkl      # Optimized model artifact
└── /outputs                    # Generated plots
    ├── profit_analysis.png     # Net Savings vs Threshold curve
    └── feature_importance.png  # Top behavioral features
```

```
```
