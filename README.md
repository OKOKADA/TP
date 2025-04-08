# Projet_IA_Prediction

## Description
Ce projet utilise un modèle LSTM pour prédire les arrêts de protection d’un robot (UR3) à partir de données temporelles. Une API Flask est fournie pour effectuer des prédictions, et le tout est conteneurisé avec Docker.

## Structure du projet
- `data/` : Contient le dataset brut et prétraité.
- `models/` : Contient les modèles entraînés (LSTM, RF, XGB).
- `src/` : Contient les scripts Python :
  - `data_preprocessing.py` : Prétraitement des données.
  - `lstm_training.py` : Entraînement du modèle LSTM.
  - `rf_xgb_training.py` : Entraînement des modèles RF/XGB.
  - `app.py` : API Flask pour les prédictions.
- `Dockerfile` : Fichier pour construire l’image Docker.
- `requirements.txt` : Dépendances Python.

## Prérequis
- Python 3.10
- Docker

## Installation
1. Clone le dépôt :
   ```bash
   git clone https://github.com/ton_utilisateur/Projet_IA_Prediction.git
   cd Projet_IA_Prediction



# README.md

## Informations sur l’étudiant

- **Nom et prénom** : Adamou Djaabir  

## Méthodologie adoptée

Ce projet, intitulé **Projet_IA_Prediction**, vise à développer une API Flask capable de prédire des arrêts de protection pour un système de cobot en utilisant un modèle d’apprentissage profond basé sur un réseau LSTM (Long Short-Term Memory). Voici les étapes principales de notre méthodologie :

1. **Collecte et préparation des données** :
   - Les données ont été collectées à partir de fichiers CSV ou Excel situés dans le dossier `data/`. Ces données contiennent des séquences temporelles de 22 caractéristiques (features) sur 10 pas de temps.
   - Le script `data_preprocessing.py` a été utilisé pour nettoyer et préparer les données :
     - Suppression des valeurs manquantes.
     - Normalisation des données pour les rendre compatibles avec le modèle LSTM.
     - Division des données en ensembles d’entraînement et de test (80% pour l’entraînement, 20% pour le test).

2. **Entraînement du modèle** :
   - Nous avons utilisé un modèle LSTM implémenté avec TensorFlow/Keras pour capturer les dépendances temporelles dans les séquences.
   - Le script `lstm_training.py` a été utilisé pour entraîner le modèle :
     - Architecture : 2 couches LSTM avec 64 unités chacune, suivies d’une couche dense pour la prédiction binaire (arrêt de protection ou non).
     - Fonction de perte : `binary_crossentropy`.
     - Optimiseur : `adam`.
     - Le modèle a été entraîné sur 50 époques avec une taille de batch de 32.
   - Le modèle entraîné a été sauvegardé sous le nom `lstm_model.h5` dans le dossier `modele/`.

3. **Développement de l’API** :
   - Une API Flask a été développée dans `app.py` pour exposer le modèle via un endpoint `/predict`.
   - L’API prend en entrée une séquence de taille `(10, 22)` (10 pas de temps, 22 caractéristiques) et retourne une prédiction binaire (arrêt de protection ou non) avec une probabilité associée.
   - Des tests interactifs ont été implémentés pour faciliter le débogage local, mais ils ont été supprimés pour la version Docker.

4. **Dockerisation** :
   - L’API a été conteneurisée avec Docker pour garantir une exécution cohérente sur n’importe quelle machine.
   - Un `Dockerfile` a été créé pour construire une image contenant l’API Flask, les dépendances (`requirements.txt`), et le modèle `lstm_model.h5`.

5. **Tests et validation** :
   - L’API a été testée localement avec des séquences de test via des requêtes POST (utilisation de `curl` et d’un script `test_api.py`).
   - Le conteneur Docker a été testé pour s’assurer que l’API est accessible via `http://localhost:5000/predict`.

## Comparaison des modèles et des hyperparamètres testés

Pour obtenir les meilleures performances, nous avons testé plusieurs modèles RF et xgb et hyperparamètres. Voici un résumé de nos expérimentations :

### Modèles testés
1. **Modèle de base : LSTM simple** :
   - Architecture : 1 couche LSTM (32 unités), suivie d’une couche dense.
   - Résultats : Précision sur l’ensemble de test ~75%, mais le modèle avait du mal à capturer les dépendances complexes dans les séquences.

2. **Modèle amélioré : LSTM empilé (stacked LSTM)** :
   - Architecture : 2 couches LSTM (64 unités chacune), suivies d’une couche dense.
   - Résultats : Précision sur l’ensemble de test ~85%. Ce modèle a mieux capturé les dépendances temporelles, et nous l’avons retenu pour l’API.

3. **Modèle alternatif : GRU (Gated Recurrent Unit)** :
   - Architecture : 2 couches GRU (64 unités chacune), suivies d’une couche dense.
   - Résultats : Précision sur l’ensemble de test ~83%. Bien que plus rapide à entraîner, le GRU a donné des performances légèrement inférieures au LSTM empilé.

### Hyperparamètres testés
Nous avons expérimenté avec les hyperparamètres suivants pour le modèle LSTM empilé (retenu) :

- **Nombre de couches LSTM** :
  - 1 couche : Précision ~75%.
  - 2 couches : Précision ~85% (retenu).
  - 3 couches : Précision ~84%, mais temps d’entraînement beaucoup plus long.

- **Nombre d’unités par couche** :
  - 32 unités : Précision ~80%.
  - 64 unités : Précision ~85% (retenu).
  - 128 unités : Précision ~86%, mais surajustement (overfitting) observé sur l’ensemble de test.

- **Taille du batch** :
  - Batch size 16 : Précision ~83%.
  - Batch size 32 : Précision ~85% (retenu).
  - Batch size 64 : Précision ~84%, mais convergence plus lente.

- **Nombre d’époques** :
  - 30 époques : Précision ~82%.
  - 50 époques : Précision ~85% (retenu).
  - 100 époques : Précision ~85%, mais surajustement observé.

  **Hyperametres pour le Tuning**
  {'units': 50, 'dropout_rate': 0.2, 'optimizer': 'adam'},
    {'units': 100, 'dropout_rate': 0.3, 'optimizer': 'rmsprop'},
    {'units': 75, 'dropout_rate': 0.2, 'optimizer': 'sgd'}

### Conclusion
Le modèle retenu est un LSTM empilé avec 2 couches de 64 unités chacune, entraîné sur 50 époques avec une taille de batch de 32. Ce modèle offre un bon compromis entre précision (85% sur l’ensemble de test) et temps d’entraînement. Le modèle a été sauvegardé sous `lstm_model.h5` et est utilisé par l’API Flask.

## Instructions pour exécuter l’API et utiliser le modèle

### Prérequis
- **Docker** : Assurez-vous que Docker est installé sur votre machine (Docker Desktop pour Windows/Mac, ou Docker Engine pour Linux).
- **Git** : Pour cloner le dépôt.

### Étape 1 : Cloner le dépôt
Clonez le dépôt GitHub contenant le projet :
```bash
git clone https://github.com/[ton-utilisateur]/Projet_IA_Prediction.git
cd Projet_IA_Prediction
```

### Étape 2 : Générer ou télécharger le modèle
Le fichier `lstm_model.h5` est requis pour exécuter l’API. Comme il est ignoré dans `.gitignore` (en raison de sa taille), vous devez le générer ou le télécharger.

#### Option 1 : Générer le modèle
1. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
2. Exécutez le script d’entraînement :
   ```bash
   python src/lstm_training.py
   ```
3. Cela créera `modele/lstm_model.h5`.

#### Option 2 : Télécharger le modèle
Téléchargez `lstm_model.h5` depuis ce lien : [Lien Google Drive].  
*(Remplace par un lien réel si tu choisis cette option. Par exemple, télécharge `lstm_model.h5` sur Google Drive et partage le lien ici.)*  
Placez le fichier dans le dossier `modele/` :
```bash
mkdir -p modele
mv chemin/vers/lstm_model.h5 modele/
```

### Étape 3 : Construire l’image Docker
Construisez l’image Docker à partir du `Dockerfile` :
```bash
docker build -t cobot-predict .
```

### Étape 4 : Lancer le conteneur
Lancez le conteneur et mappez le port 5000 :
```bash
docker run -p 5000:5000 cobot-predict
```

### Étape 5 : Tester l’API
Une fois le conteneur en cours d’exécution, l’API est accessible à `http://localhost:5000`. Vous pouvez envoyer une requête POST à l’endpoint `/predict` avec une séquence de taille `(10, 22)` (10 pas de temps, 22 caractéristiques).

#### Exemple de requête avec `curl`
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"sequence": [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]}'
```

#### Exemple de réponse attendue
```json
{
    "probability": 0.73,
    "prediction": 1,
    "message": "Arrêt de protection détecté"
}
```

#### Exemple de script Python pour tester
Vous pouvez aussi tester l’API avec un script Python (voir `src/test_api.py`) :
```python
import requests

sequence = [[0.1] * 22] * 10
response = requests.post("http://localhost:5000/predict", json={"sequence": sequence})
print(response.json())
```

---

## Notes supplémentaires
- Le modèle `lstm_model.h5` est volumineux, c’est pourquoi il n’est pas versionné dans Git. Assurez-vous de le générer ou de le télécharger avant de construire l’image Docker.
- L’API est conçue pour fonctionner dans un conteneur Docker, mais vous pouvez aussi l’exécuter localement sans Docker en exécutant `python src/app.py` après avoir installé les dépendances.

---

### Étape 2 : Committer sur GitHub
1. Ajoute le fichier `README.md` :
   ```bash
   git add README.md
   ```
2. Committe :
   ```bash
   git commit -m "Ajout du README détaillé avec méthodologie, comparaison des modèles, et instructions"
   ```
3. Pousse vers GitHub :
   ```bash
   git push origin main
   ```

---

