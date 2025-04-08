# src/app.py
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

# Initialisation de l'application Flask
app = Flask(__name__)

# Charger le modèle Keras
model = load_model("lstm_model.h5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.eval = lambda: None

# Paramètres de la séquence
seq_length = 10
input_size = 22

# Endpoint pour la prédiction
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

if __name__ == '_main_':
    app.run(host='0.0.0.0', port=5000, debug=True)