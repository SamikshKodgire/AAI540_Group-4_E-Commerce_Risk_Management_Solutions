#!/usr/bin/env python3
import os
import json
import tarfile
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main():
    base_dir = '/opt/ml/processing'
    model_path = os.path.join(base_dir, 'model', 'model.tar.gz')
    test_path = os.path.join(base_dir, 'test', 'test.csv')
    output_dir = os.path.join(base_dir, 'evaluation')
    
    print("Extracting model...")
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    print("Loading model...")
    # SageMaker built-in XGBoost saves the model as 'xgboost-model'
    model = xgb.Booster()
    model.load_model("xgboost-model")
    
    print("Loading test data...")
    # Read test data (no headers, target is the first column)
    test_df = pd.read_csv(test_path, header=None)
    y_test = test_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values
    
    dtest = xgb.DMatrix(X_test)
    
    print("Generating predictions...")
    predictions_proba = model.predict(dtest)
    # Default threshold of 0.5 for binary classification
    predictions = np.where(predictions_proba > 0.5, 1, 0)
    
    print("Calculating metrics...")
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, predictions_proba)
    
    report_dict = {
        "classification_metrics": {
            "accuracy": {"value": accuracy},
            "precision": {"value": precision},
            "recall": {"value": recall},
            "f1_score": {"value": f1},
            "roc_auc": {"value": roc_auc}
        }
    }
    
    print("Saving evaluation report...")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        f.write(json.dumps(report_dict))
        
    print(f"Evaluation complete. ROC-AUC: {roc_auc:.4f}, F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()
