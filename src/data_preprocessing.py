# src/data_preprocessing.py
import pandas as pd
import numpy as np

# Charger le dataset
df = pd.read_excel("../data/dataset_02052023.xlsx")
print("Dataset chargé :", df.shape)

# Prétraitement (exemple simplifié)
df['Timestamp'] = pd.to_datetime(df['Timestamp'].astype(str).str.replace('"', ''), format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')
df = df.dropna(subset=['Timestamp'])
df = df.sort_values(by='Timestamp')
df['Timestamp'] = (df['Timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
df['grip_lost'] = df['grip_lost'].astype(int)
df['Robot_ProtectiveStop'] = df['Robot_ProtectiveStop'].astype(int)

# Sauvegarder le DataFrame prétraité
df.to_csv("../data/dataset_preprocessed.csv", index=False)
print("DataFrame prétraité sauvegardé sous '../data/dataset_preprocessed.csv'")
