import time
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.neighbors import KNeighborsRegressor

# Excel-Datei laden
file_path = 'C:\\Users\\fbrze\\Documents\\EigeneDateien\\Programmierung_etc\\Git\\studium-aufgaben\\Sem5\\ILS_Praktikum\\versuch3\\V3A1\\airfoil_self_noise.xls'

# Daten aus Excel-Datei laden
df = pd.read_excel(file_path, 0)

# Features und Zielvariable auswählen
X = df.iloc[:, 0:5].values  # Features als 2D-Array
y = df.iloc[:, 5].values

# MLP Regressor erstellen
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42, max_iter=1000)

# Kreuzvalidierung durchführen für MLP
kf = KFold(n_splits=3, shuffle=True, random_state=42)
mae_scorer = make_scorer(mean_absolute_error)
start_time = time.time() # Zeitmessung
scoresMLP = cross_val_score(mlp, X, y, cv=kf, scoring=mae_scorer)
mlp_trainingtime = time.time() - start_time

# Kreuzvalidierung durchführen für KNN
knn = KNeighborsRegressor(n_neighbors=5, weights='uniform')
start_time = time.time() # Zeitmessung
scoresKNN = cross_val_score(knn, X, y, cv=kf, scoring=mae_scorer)
knn_trainingtime = time.time() - start_time




# Ergebnisse anzeigen
print("[MLP] Mean Absolute Error pro Fold:", scoresMLP)
print("[MLP] Durchschnittlicher Mean Absolute Error:", scoresMLP.mean())
print("[MLP] Standardabweichung des Mean Absolute Error:", scoresMLP.std())
print("[MLP] Dauer:", mlp_trainingtime)
print("===============================")

print("[KNN] Mean Absolute Error pro Fold:", scoresKNN)
print("[KNN] Durchschnittlicher Mean Absolute Error:", scoresKNN.mean())
print("[KNN] Standardabweichung des Mean Absolute Error:", scoresKNN.std())
print("[KNN] Dauer:", knn_trainingtime)
