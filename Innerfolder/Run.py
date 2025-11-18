# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 21:03:23 2025

@author: ajp07
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from Smiles import generate_maccs_keys, generate_morgan_fingerprints

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def run_analysis():
    df = pd.read_csv("lipophilicity.csv") 
    smiles = df['smiles'].tolist()
    y = df['exp'].values 

    maccs_fps = generate_maccs_keys(smiles)
    morgan_fps = generate_morgan_fingerprints(smiles)

    X_maccs = np.array([list(fp) for fp in maccs_fps])
    X_morgan = np.array([list(fp) for fp in morgan_fps])

    X_train_maccs, X_test_maccs, y_train, y_test = train_test_split(X_maccs, y, test_size=0.2, random_state=42)
    X_train_morgan, X_test_morgan, _, _ = train_test_split(X_morgan, y, test_size=0.2, random_state=42)

    model_maccs = LinearRegression().fit(X_train_maccs, y_train)
    model_morgan = LinearRegression().fit(X_train_morgan, y_train)

    pred_maccs = model_maccs.predict(X_test_maccs)
    rmse_maccs = calculate_rmse(y_test, pred_maccs)

    pred_morgan = model_morgan.predict(X_test_morgan)
    rmse_morgan = calculate_rmse(y_test, pred_morgan)

    conda_env = os.getenv("CONDA_DEFAULT_ENV")

    print("-" * 40)
    print("Lipophilicity Model Results")
    print("-" * 40)
    print(f"MACCS Key Model RMSE: {rmse_maccs:.4f}")
    print(f"Morgan Fingerprint Model RMSE: {rmse_morgan:.4f}")
    print(f"\nConda Environment Used: {conda_env}")
    print("-" * 40)

if __name__ == "__main__":
    run_analysis()

