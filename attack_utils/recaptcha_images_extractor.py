import os
import shutil
import urllib.request
import zipfile

# Lien vers le référentiel GitHub
github_repo_url = 'https://github.com/folfcoder/recaptcha-dataset/archive/refs/heads/main.zip'

# Téléchargement et extraction du fichier ZIP du référentiel
zip_file_path = 'data.zip'
urllib.request.urlretrieve(github_repo_url, zip_file_path)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('data')

# Déplacement des données d'entraînement dans un dossier spécifique
data_dir = 'data/recaptcha-dataset-main/Large'
train_dir = 'train_data'

os.makedirs(train_dir, exist_ok=True)
shutil.move(data_dir, train_dir)

# Suppression du fichier ZIP après extraction
os.remove(zip_file_path)
