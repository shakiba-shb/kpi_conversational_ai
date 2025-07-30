# This script contains mcp tools for the chatbot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from langchain.tools import tool
from typing import Optional


# Load preprocessed data
DATA_PATH = "data/KPI_Anomaly_Detection/data/AD_data_10KPI.csv"
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])

# 1. Detect anomalies using IForest, KNN, and PCA (consensus voting)
def detect_anomalies(kpi: str, site_id: str, start_date: str = None, end_date: str = None) -> str:
    """Detect anomalies in a specific KPI for a given sector and date range using IForest, KNN, and PCA consensus."""
    data = df[df['Sector_ID'] == site_id].copy()
    if start_date:
        data = data[data['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        data = data[data['Date'] <= pd.to_datetime(end_date)]

    if kpi not in data.columns:
        return f"KPI '{kpi}' not found."
    if data.empty:
        return f"No data available for {site_id} and KPI '{kpi}' between {start_date} and {end_date}."
    
    values = data[[kpi]].values

    models = {
        'IForest': IForest(contamination=0.1),
        'KNN': KNN(contamination=0.1),
        'PCA': PCA(contamination=0.1)
    }

    predictions = []
    for name, model in models.items():
        model.fit(values)
        pred = model.predict(values)  # 1 for outlier, 0 for inlier
        predictions.append(pred)

    votes = np.sum(predictions, axis=0)
    consensus_mask = votes >= 2
    anomalies = data[consensus_mask]

    if anomalies.empty:
        return f"No consensus anomalies found in {kpi} for {site_id}."

    result = anomalies[['Date', kpi]].to_string(index=False)
    return f"Consensus anomalies (voted by >=2 models) in {kpi} for {site_id}:{result}"


# 2. Generate and save a plot of KPI anomalies
def plot_kpi_anomalies(kpi: str, site_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """
    Generate and save a time series plot of a KPI for a specific site,
    highlighting anomalies flagged by majority vote from multiple models.

    Args:
        kpi (str): The KPI name (e.g., 'RTT', 'Packet_Loss')
        site_id (str): The sector ID (e.g., 'SITE_001_SECTOR_A')
        start_date (str, optional): Start date (YYYY-MM-DD)
        end_date (str, optional): End date (YYYY-MM-DD)

    Returns:
        str: File path or error message
    """
    try:
        filtered = df[df['Sector_ID'] == site_id].copy()

        if start_date:
            filtered = filtered[filtered['Date'] >= start_date]
        if end_date:
            filtered = filtered[filtered['Date'] <= end_date]

        if filtered.empty:
            return f"No data available for {site_id} and KPI '{kpi}' in the specified date range."

        if kpi not in filtered.columns:
            return f"KPI '{kpi}' not found in the dataset."

        filtered = filtered.sort_values('Date')
        filtered['Date'] = pd.to_datetime(filtered['Date'])

        values = filtered[[kpi]].values

        models = {
            'IForest': IForest(contamination=0.1),
            'KNN': KNN(contamination=0.1),
            'PCA': PCA(contamination=0.1)
        }

        predictions = []
        for model in models.values():
            model.fit(values)
            pred = model.predict(values)
            predictions.append(pred)

        votes = np.sum(predictions, axis=0)
        anomaly_mask = votes >= 2

        plt.figure(figsize=(10, 4))
        plt.plot(filtered['Date'], filtered[kpi], label=kpi, color='blue')
        plt.scatter(filtered['Date'][anomaly_mask], filtered[kpi][anomaly_mask], color='red', label='Anomalies')
        plt.xlabel("Date")
        plt.ylabel(kpi)
        plt.title(f"{kpi} Time Series with Anomalies for {site_id}")
        plt.legend()

        os.makedirs("outputs", exist_ok=True)
        output_path = f"/home/shakiba/KPI_Anomaly_Detection/outputs/{site_id}_{kpi}_anomalies.png"
        plt.savefig(output_path)
        plt.close()

        return output_path

    except Exception as e:
        return f"Error during plotting: {str(e)}"