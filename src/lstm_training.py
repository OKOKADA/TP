# src/app.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import requests
import threading
import json

app = Flask(__name__)

model = load_model("../modele/lstm_model.h5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Évaluer immédiatement après la compilation
print("Évaluation du modèle sur des données factices pour construire les métriques...")
dummy_input = np.random.random((1, 10, 22))
dummy_output = np.array([0])
model.evaluate(dummy_input, dummy_output, verbose=0)
print("Évaluation terminée.")

model.eval = lambda: None

seq_length = 10
input_size = 22

# Reste du code inchangé...
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'sequence' not in data:
            return jsonify({"error": "Aucune séquence fournie"}), 400

        sequence = np.array(data['sequence'], dtype=np.float32)
        if sequence.shape != (seq_length, input_size):
            return jsonify({"error": f"La séquence doit être de taille ({seq_length}, {input_size})"}), 400

        sequence_array = sequence.reshape(1, seq_length, input_size)
        output = model.predict(sequence_array, verbose=0)[0][0]
        prediction = 1 if output > 0.5 else 0

        return jsonify({
            "probability": float(output),
            "prediction": int(prediction),
            "message": "Arrêt de protection détecté" if prediction == 1 else "Pas d'arrêt de protection"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['GET'])
def predict_get():
    return jsonify({"message": "Utilisez une requête POST avec une séquence de données pour obtenir une prédiction."})

@app.route('/favicon.ico')
def favicon():
    return '', 204

def run_app():
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

def interactive_test():
    print("\n=== Test interactif de l’API ===")
    print("Entrez une séquence de 10 pas temporels avec 22 features.")
    print("Format attendu : [[valeur1, valeur2, ..., valeur22], ..., [valeur1, valeur2, ..., valeur22]]")
    print("Exemple : [[0.1, 0.1, ..., 0.1], ..., [0.1, 0.1, ..., 0.1]]")
    print("Tapez 'exit' pour quitter.\n")

    while True:
        user_input = input("Entrez votre séquence (ou 'exit' pour quitter) : ")
        if user_input.lower() == 'exit':
            print("Arrêt du test interactif.")
            break

        try:
            sequence = json.loads(user_input)
            sequence = np.array(sequence, dtype=np.float32)
            if sequence.shape != (seq_length, input_size):
                print(f"Erreur : La séquence doit être de taille ({seq_length}, {input_size}).")
                continue

            response = requests.post("http://localhost:5000/predict", json={"sequence": sequence.tolist()})
            response.raise_for_status()

            print("\nRésultat de la prédiction :")
            print(json.dumps(response.json(), indent=4))
            print()

        except json.JSONDecodeError:
            print("Erreur : Format de séquence invalide. Assurez-vous d’entrer une liste JSON valide.")
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête : {e}")
        except Exception as e:
            print(f"Erreur : {e}")

if __name__ == '__main__':
    server_thread = threading.Thread(target=run_app)
    server_thread.daemon = True
    server_thread.start()

    import time
    time.sleep(2)

    interactive_test()