# Utiliser une image de base Python 3.10
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
COPY src/ src/
COPY modele/ modele/

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port 5000 (utilisé par Flask)
EXPOSE 5000

# Commande pour lancer l’API
CMD ["python", "src/app.py"]