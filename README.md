import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Chargement des données à partir d'un fichier CSV
data = pd.read_csv('train.csv')

# Affichage des premières lignes des données
print(data.head())

# Sélection des caractéristiques (features) et de la cible (target)
caracteristiques = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
cible = data['SalePrice']

# Remplissage des valeurs manquantes avec la moyenne des caractéristiques
caracteristiques.fillna(caracteristiques.mean(), inplace=True)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(caracteristiques, cible, test_size=0.2, random_state=42)

# Création et entraînement du modèle de régression linéaire
modele = LinearRegression()
modele.fit(X_train, y_train)

# Prédiction des valeurs cibles pour l'ensemble de test
y_pred = modele.predict(X_test)

# Calcul de l'erreur quadratique moyenne et du coefficient de détermination (R2)
erreur_quadratique_moyenne = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Erreur Quadratique Moyenne: {erreur_quadratique_moyenne}')
print(f'Coefficient de Détermination (R²): {r2}')

# Visualisation des prix réels vs prédits
plt.scatter(y_test, y_pred)
plt.xlabel('Prix Réels')
plt.ylabel('Prix Prédits')
plt.title('Prix Réels vs Prix Prédits')
plt.show()
