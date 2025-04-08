# src/test_api.py
import requests

# Exemple de séquence (10 pas temporels, 22 features)
sequence = [[0.1] * 22] * 10

# Envoyer la requête POST à l’API
response = requests.post("http://localhost:5000/predict", json={"sequence": sequence})
print(response.json())