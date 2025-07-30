# This script converts a CSV data file to an SQLite database.
import pandas as pd
import sqlite3
import os

csv_path = "data/AD_data_10KPI.csv"
db_path = "data/kpi_dataset.db"
table_name = "kpi_data"

df = pd.read_csv(csv_path)

# Convert 'Date' column to datetime format 
if not pd.api.types.is_datetime64_any_dtype(df['Date']):
    df['Date'] = pd.to_datetime(df['Date'])

conn = sqlite3.connect(db_path)

df.to_sql(table_name, conn, if_exists='replace', index=False)
print(f" KPI data written to {db_path} as table '{table_name}'")

conn.close()